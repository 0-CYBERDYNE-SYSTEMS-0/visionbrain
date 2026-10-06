"""Task-only Watch changes preserve the active lease deadline."""

import asyncio
import base64
import hashlib
import threading
import time
from dataclasses import asdict
from io import BytesIO

import pytest
from PIL import Image

from visionbrain.mission_contracts import (
    Decision,
    FindingProposal,
    GeometryItem,
    Principal,
    SourceBinding,
    SourceFrame,
    ToolResult,
    WatchLease,
    WatchLeaseRelease,
    WatchProposal,
)
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (32, 24), (40, 90, 120)).save(output, format="JPEG")
    return output.getvalue()


class _Source:
    def __init__(self, binding):
        self.binding = binding
        self.frame_id = 0

    def __call__(self, binding):
        self.frame_id += 1
        return SourceFrame(
            _jpeg(), binding.source_id, binding.source_epoch, self.frame_id, time.monotonic()
        )


class _Planner:
    def __init__(self):
        self.calls = 0

    def available(self, _model_key):
        return True

    def plan(self, _context):
        self.calls += 1
        watch = WatchProposal(("container",), "detect") if self.calls == 1 else None
        return Decision(1, "finish", watch=watch)

    def close(self):
        return None


class _Tools:
    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, _request, _context):
        raise AssertionError("finish should not invoke a perception tool")


class _EvidencePlanner:
    def __init__(self):
        self.calls = 0

    def available(self, _model_key):
        return True

    def plan(self, context):
        self.calls += 1
        if not context.tool_results:
            return Decision(1, "detect_objects", {"targets": ["label"]})
        if context.tool_results[-1].tool == "detect_objects":
            return Decision(1, "read_text", {"item_id": context.tool_results[-1].items[0].item_id})
        detected, ocr = context.tool_results[-2:]
        return Decision(
            1,
            "finish",
            findings=(
                FindingProposal(
                    claim="LABEL-17",
                    claim_type="text_read",
                    evidence_refs=(context.input_evidence_id,),
                    text_refs=(ocr.tool_result_id,),
                ),
                FindingProposal(
                    claim="The label is damaged.",
                    claim_type="localized_object",
                    evidence_refs=(context.input_evidence_id,),
                    item_refs=((detected.tool_result_id, detected.items[0].item_id),),
                ),
            ),
        )

    def close(self):
        return None


class _EvidenceTools:
    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, request, _context):
        if request.tool == "detect_objects":
            return ToolResult(
                "ok", items=(GeometryItem("label-1", "label", 0.9, (0.1, 0.2, 0.7, 0.8)),)
            )
        return ToolResult("ok", text="LABEL-17")


class _Watch:
    def __init__(self):
        self.active = None
        self.applied = []
        self.released = []
        self.release_results = []
        self.revision = 0
        self.returned_mission_id = None

    def current_configuration_revision(self):
        return self.revision

    def apply(self, request):
        prior = self.active
        if request.expected_configuration_revision != self.revision:
            raise RuntimeError("configuration_revision_conflict")
        if prior is not None and (
            prior.mission_id != request.mission_id
            or prior.source_binding != request.source_binding
            or prior.configuration_revision != request.expected_configuration_revision
        ):
            raise RuntimeError("watch_lease_active")
        changed = prior is None or prior.targets != request.targets or prior.task != request.task
        if prior is None or changed:
            self.revision += 1
        self.active = WatchLease(
            lease_id=prior.lease_id if prior else "task-lease-1",
            mission_id=self.returned_mission_id or request.mission_id,
            source_binding=request.source_binding,
            configuration_revision=self.revision,
            targets=request.targets,
            task=request.task,
            expires_at_ms=request.expires_at_ms,
        )
        self.applied.append(self.active)
        return self.active

    def release(self, lease, reason):
        self.released.append((lease, reason))
        matches = self.active is not None and (
            self.active.lease_id == lease.lease_id
            and self.active.mission_id == lease.mission_id
            and self.active.source_binding == lease.source_binding
            and self.active.configuration_revision == lease.configuration_revision
            and self.active.expires_at_ms == lease.expires_at_ms
        )
        if matches:
            self.active = None
            self.revision += 1
            result = WatchLeaseRelease(True, self.revision)
        else:
            result = WatchLeaseRelease(False, self.revision, "lease_not_active")
        self.release_results.append(result)
        return result


def _command(command, request_id, *, mission_id=None, revision=None, args=None):
    message = {
        "type": "mission_command",
        "schema_version": 1,
        "request_id": request_id,
        "command": command,
        "mission_id": mission_id,
        "args": args or {},
    }
    if revision is not None:
        message["expected_revision"] = revision
    return message


async def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


async def _active_watch(runtime, store, binding):
    prepared = await runtime.handle(
        _command(
            "prepare",
            "prepare-rollback-watch",
            args={
                "profile": {"id": "visual_inspection", "version": 1},
                "expertise": "facility reviewer",
                "goal": "Inspect visible conditions",
                "mode": "watch",
                "reasoning_model": "gemma",
                "source_id": binding.source_id,
            },
        ),
        Principal("operator"),
        SCOPES,
        integrated=True,
    )
    snapshot = prepared["result"]["snapshot"]
    await runtime.handle(
        _command(
            "activate",
            "activate-rollback-watch",
            mission_id=snapshot["mission_id"],
            revision=snapshot["revision"],
            args={"source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch}},
        ),
        Principal("operator"),
        SCOPES,
        integrated=True,
    )
    current = await _wait_for(
        lambda: (
            value
            if (value := store.get_mission(snapshot["mission_id"]))
            and value.get("watch_lease")
            and value.get("cycle_history")
            else None
        )
    )
    return current


def _inject_update_commit_failure(monkeypatch, store, request_id, entered, allow_failure):
    original = store.perform_request

    def perform_request(principal_id, current_id, payload, operation, *, now_ms):
        if current_id != request_id:
            return original(principal_id, current_id, payload, operation, now_ms=now_ms)

        def fail_after_operation(tx):
            operation(tx)
            entered.set()
            if not allow_failure.wait(3):
                raise TimeoutError("commit failure barrier timed out")
            raise OSError("injected SQLite commit failure")

        return original(principal_id, current_id, payload, fail_after_operation, now_ms=now_ms)

    monkeypatch.setattr(store, "perform_request", perform_request)


def _inject_update_commit_barrier(
    monkeypatch, store, request_id, entered, allow_commit, *, failure
):
    original = store.perform_request

    def perform_request(principal_id, current_id, payload, operation, *, now_ms):
        if current_id != request_id:
            return original(principal_id, current_id, payload, operation, now_ms=now_ms)

        def wait_before_commit(tx):
            reply = operation(tx)
            entered.set()
            if not allow_commit.wait(3):
                raise TimeoutError("commit barrier timed out")
            if failure:
                raise OSError("injected SQLite commit failure")
            return reply

        return original(principal_id, current_id, payload, wait_before_commit, now_ms=now_ms)

    monkeypatch.setattr(store, "perform_request", perform_request)


def test_task_only_mask_updates_live_task_without_extending_null_proposal_lease(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.03)
    monkeypatch.setattr(runtime_module, "WATCH_LEASE_SECONDS", 0.5)
    binding = SourceBinding("task-lease-source", "1")
    source, planner, watch = _Source(binding), _Planner(), _Watch()
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    runtime = MissionRuntime(
        store,
        planner,
        _Tools(),
        source,
        watch,
        lambda _event: None,
        qualified_models={"gemma": ["watch"]},
        approved_source_ids={binding.source_id},
    )

    async def scenario():
        try:
            prepared = await runtime.handle(
                _command(
                    "prepare",
                    "prepare-task-lease",
                    args={
                        "profile": {"id": "visual_inspection", "version": 1},
                        "expertise": "facility reviewer",
                        "goal": "Inspect visible conditions",
                        "mode": "watch",
                        "reasoning_model": "gemma",
                        "source_id": binding.source_id,
                    },
                ),
                Principal("operator"),
                SCOPES,
                integrated=True,
            )
            snapshot = prepared["result"]["snapshot"]
            activated = await runtime.handle(
                _command(
                    "activate",
                    "activate-task-lease",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch}},
                ),
                Principal("operator"),
                SCOPES,
                integrated=True,
            )
            assert activated["ok"]
            mission_id = snapshot["mission_id"]
            first_cycle = await _wait_for(
                lambda: (
                    current
                    if (current := store.get_mission(mission_id))
                    and current.get("watch_lease")
                    and current.get("cycle_history")
                    else None
                )
            )
            initial_expiry = first_cycle["watch_lease"]["expires_at_ms"]

            updated = await runtime.handle(
                _command(
                    "update_brief",
                    "mask-current-lease",
                    mission_id=mission_id,
                    revision=first_cycle["revision"],
                    args={"watch_task": "segment"},
                ),
                Principal("operator"),
                SCOPES,
                integrated=True,
            )
            assert updated["ok"]
            current = updated["result"]["snapshot"]
            assert current["state"] == "running"
            assert current["watch_task"] == "segment"
            assert current["task"] == "segment"
            assert current["watch_lease"]["task"] == "segment"
            assert current["watch_lease"]["expires_at_ms"] == initial_expiry
            assert len(watch.applied) == 2

            second_cycle = await _wait_for(
                lambda: (
                    latest
                    if (latest := store.get_mission(mission_id))
                    and len(latest.get("cycle_history", ())) >= 2
                    else None
                )
            )
            assert len(watch.applied) == 2
            assert second_cycle["watch_lease"]["expires_at_ms"] == initial_expiry

            expired = await _wait_for(
                lambda: (
                    latest
                    if (latest := store.get_mission(mission_id))
                    and latest.get("reason") == "watch_lease_expired"
                    else None
                ),
                timeout=2.0,
            )
            assert expired["state"] == "paused"
            assert len(watch.applied) == 2
            await _wait_for(lambda: watch.released)
            assert len(watch.released) == 1
            assert watch.released[0][1] == "watch_lease_expired"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("manual_takeover", [False, True])
def test_task_only_update_commit_failure_releases_only_its_exact_applied_lease(
    tmp_path, monkeypatch, manual_takeover
):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 2.0)
    monkeypatch.setattr(runtime_module, "WATCH_LEASE_SECONDS", 30.0)
    binding = SourceBinding("rollback-source", "1")
    watch = _Watch()
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    runtime = MissionRuntime(
        store,
        _Planner(),
        _Tools(),
        _Source(binding),
        watch,
        lambda _event: None,
        qualified_models={"gemma": ["watch"]},
        approved_source_ids={binding.source_id},
    )

    async def scenario():
        entered, allow_failure = threading.Event(), threading.Event()
        try:
            initial = await _active_watch(runtime, store, binding)
            mission_id = initial["mission_id"]
            persisted_lease = initial["watch_lease"]
            prior_timer = runtime._lease_timers[mission_id]
            _inject_update_commit_failure(
                monkeypatch, store, "rollback-task-update", entered, allow_failure
            )
            command = _command(
                "update_brief",
                "rollback-task-update",
                mission_id=mission_id,
                revision=initial["revision"],
                args={"watch_task": "segment"},
            )
            task = asyncio.create_task(
                runtime.handle(command, Principal("operator"), SCOPES, integrated=True)
            )
            try:
                assert await asyncio.to_thread(entered.wait, 2.0)
                applied = watch.applied[-1]
                assert applied.task == "segment"
                if manual_takeover:
                    manual = WatchLease(
                        lease_id="manual-owner",
                        mission_id=mission_id,
                        source_binding=binding,
                        configuration_revision=watch.revision + 1,
                        targets=("manual target",),
                        task="detect",
                        expires_at_ms=applied.expires_at_ms + 10_000,
                    )
                    watch.revision = manual.configuration_revision
                    watch.active = manual
                allow_failure.set()
                with pytest.raises(OSError, match="injected SQLite commit failure"):
                    await task
            finally:
                allow_failure.set()
                if not task.done():
                    try:
                        await task
                    except OSError:
                        pass

            current = store.get_mission(mission_id)
            assert current["watch_lease"] == persisted_lease
            assert current.get("watch_task") == initial.get("watch_task")
            assert watch.released[-1] == (applied, "watch_task_update_not_committed")
            assert watch.release_results[-1].released is (not manual_takeover)
            if manual_takeover:
                assert watch.active.lease_id == "manual-owner"
            else:
                assert watch.active is None
            assert runtime._lease_timers[mission_id] is prior_timer
            assert not prior_timer.cancelled()
            memo = store._connection.execute(
                "SELECT 1 FROM requests WHERE principal_id = ? AND request_id = ?",
                ("operator", "rollback-task-update"),
            ).fetchone()
            assert memo is None
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", [False, True])
def test_repeated_cancel_waits_for_task_lease_transaction_before_cleanup(
    tmp_path, monkeypatch, failure
):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 2.0)
    monkeypatch.setattr(runtime_module, "WATCH_LEASE_SECONDS", 30.0)
    binding = SourceBinding("cancel-rollback-source", "1")
    watch = _Watch()
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    runtime = MissionRuntime(
        store,
        _Planner(),
        _Tools(),
        _Source(binding),
        watch,
        lambda _event: None,
        qualified_models={"gemma": ["watch"]},
        approved_source_ids={binding.source_id},
    )

    async def scenario():
        entered, allow_commit = threading.Event(), threading.Event()
        task = None
        try:
            initial = await _active_watch(runtime, store, binding)
            mission_id = initial["mission_id"]
            prior_timer = runtime._lease_timers[mission_id]
            _inject_update_commit_barrier(
                monkeypatch,
                store,
                "cancel-task-update",
                entered,
                allow_commit,
                failure=failure,
            )
            task = asyncio.create_task(
                runtime.handle(
                    _command(
                        "update_brief",
                        "cancel-task-update",
                        mission_id=mission_id,
                        revision=initial["revision"],
                        args={"watch_task": "segment"},
                    ),
                    Principal("operator"),
                    SCOPES,
                    integrated=True,
                )
            )
            assert await asyncio.to_thread(entered.wait, 2.0)
            applied = watch.applied[-1]
            task.cancel()
            await asyncio.sleep(0.01)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert watch.active is applied
            assert watch.released == []

            allow_commit.set()
            if failure:
                with pytest.raises(OSError, match="injected SQLite commit failure"):
                    await task
                current = store.get_mission(mission_id)
                assert current["watch_lease"] == initial["watch_lease"]
                assert current.get("watch_task") == initial.get("watch_task")
                assert watch.released[-1] == (
                    applied,
                    "watch_task_update_not_committed",
                )
                assert watch.release_results[-1].released
                assert watch.active is None
                assert runtime._lease_timers[mission_id] is prior_timer
                assert not prior_timer.cancelled()
            else:
                with pytest.raises(asyncio.CancelledError):
                    await task
                current = store.get_mission(mission_id)
                saved_lease = current["watch_lease"]
                assert saved_lease["lease_id"] == applied.lease_id
                assert saved_lease["configuration_revision"] == applied.configuration_revision
                assert saved_lease["expires_at_ms"] == applied.expires_at_ms
                assert saved_lease["task"] == applied.task
                assert current["watch_task"] == "segment"
                assert watch.active is applied
                assert watch.released == []
                assert runtime._lease_timers[mission_id] is not prior_timer
                assert not runtime._lease_timers[mission_id].cancelled()
                assert prior_timer.cancelled()
        finally:
            allow_commit.set()
            if task is not None and not task.done():
                try:
                    await task
                except BaseException:
                    pass
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_invalid_returned_task_lease_is_released_and_mission_identity_checked(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 2.0)
    monkeypatch.setattr(runtime_module, "WATCH_LEASE_SECONDS", 30.0)
    binding = SourceBinding("invalid-lease-source", "1")
    watch = _Watch()
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    runtime = MissionRuntime(
        store,
        _Planner(),
        _Tools(),
        _Source(binding),
        watch,
        lambda _event: None,
        qualified_models={"gemma": ["watch"]},
        approved_source_ids={binding.source_id},
    )

    async def scenario():
        try:
            initial = await _active_watch(runtime, store, binding)
            watch.returned_mission_id = "wrong-mission"
            rejected = await runtime.handle(
                _command(
                    "update_brief",
                    "invalid-task-lease",
                    mission_id=initial["mission_id"],
                    revision=initial["revision"],
                    args={"watch_task": "segment"},
                ),
                Principal("operator"),
                SCOPES,
                integrated=True,
            )
            assert not rejected["ok"]
            assert rejected["error"]["code"] == "watch_lease_invalid"
            assert watch.released[-1] == (watch.applied[-1], "watch_task_update_not_committed")
            assert watch.release_results[-1].released
            assert watch.active is None
            current = store.get_mission(initial["mission_id"])
            assert current["watch_lease"] == initial["watch_lease"]
            assert current.get("watch_task") == initial.get("watch_task")
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_cycle_commit_merges_tombstone_into_new_findings_and_localization(tmp_path, monkeypatch):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    runtime = MissionRuntime(
        store,
        _EvidencePlanner(),
        _EvidenceTools(),
        lambda _binding: None,
        _Watch(),
        lambda _event: None,
        qualified_models={"gemma": ["inspect"]},
    )

    async def scenario():
        entered, allow_commit = threading.Event(), threading.Event()
        original_change = runtime._change_snapshot
        held = {}

        async def hold_cycle_commit(mission_id, generation, updated, kind, data):
            if kind == "cycle_finished" and updated.get("findings") and not held:
                held["updated"] = dict(updated)
                entered.set()
                if not await asyncio.to_thread(allow_commit.wait, 3.0):
                    raise TimeoutError("cycle commit barrier timed out")
            return await original_change(mission_id, generation, updated, kind, data)

        runtime._change_snapshot = hold_cycle_commit
        try:
            created = await runtime.handle(
                _command(
                    "create",
                    "create-cycle-tombstone",
                    args={
                        "profile": {"id": "visual_inspection", "version": 1},
                        "expertise": "facility reviewer",
                        "goal": "Inspect visible conditions",
                        "mode": "inspect",
                        "reasoning_model": "gemma",
                    },
                ),
                Principal("operator"),
                SCOPES,
            )
            snapshot = created["result"]["snapshot"]
            attached = await runtime.handle(
                _command(
                    "attach_evidence",
                    "attach-cycle-tombstone",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={
                        "jpeg_b64": base64.b64encode(_jpeg()).decode("ascii"),
                        "sha256": hashlib.sha256(_jpeg()).hexdigest(),
                    },
                ),
                Principal("operator"),
                SCOPES,
            )
            snapshot = attached["result"]["snapshot"]
            await runtime.handle(
                _command(
                    "resume",
                    "resume-cycle-tombstone",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                ),
                Principal("operator"),
                SCOPES,
            )
            signalled = await asyncio.to_thread(entered.wait, 3.0)
            assert signalled, store.planner_diagnostics(snapshot["mission_id"])
            mission_id = snapshot["mission_id"]
            input_evidence_id = snapshot["input_evidence_id"]
            assert all(
                finding["evidence_id"] == input_evidence_id
                for finding in held["updated"]["findings"]
            )
            await asyncio.to_thread(runtime._mark_evidence_unavailable, mission_id, input_evidence_id)
            allow_commit.set()

            committed = await _wait_for(
                lambda: (
                    current
                    if (current := store.get_mission(mission_id))
                    and len(current.get("cycle_history", ())) >= 1
                    else None
                )
            )
            evidence = next(
                row for row in committed["evidence"] if row["evidence_id"] == input_evidence_id
            )
            assert evidence["available"] is False
            text_finding = next(row for row in committed["findings"] if row["claim"] == "LABEL-17")
            assert text_finding["status"] == "unresolved"
            assert text_finding["reason"] == "evidence_unavailable"
            assert text_finding["text_refs"]
            localized = next(
                row for row in committed["findings"] if row["claim"] == "The label is damaged."
            )
            assert localized["status"] == "unresolved"
            assert localized["reason"] == "geometry_does_not_prove_semantic_claim"
            assert localized["items"][0]["item_id"] == "label-1"
            assert localized["localization"]["status"] == "unresolved"
            assert localized["localization"]["reason"] == "evidence_unavailable"
            assert localized["localization"]["statements"][0]["label"] == "label"
        finally:
            allow_commit.set()
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_task_only_update_does_not_apply_an_expired_lease(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 2.0)
    monkeypatch.setattr(runtime_module, "WATCH_LEASE_SECONDS", 30.0)
    binding = SourceBinding("expired-lease-source", "1")
    watch = _Watch()
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    runtime = MissionRuntime(
        store,
        _Planner(),
        _Tools(),
        _Source(binding),
        watch,
        lambda _event: None,
        qualified_models={"gemma": ["watch"]},
        approved_source_ids={binding.source_id},
    )

    async def scenario():
        try:
            initial = await _active_watch(runtime, store, binding)
            mission_id = initial["mission_id"]
            expired = WatchLease(
                lease_id=watch.active.lease_id,
                mission_id=watch.active.mission_id,
                source_binding=watch.active.source_binding,
                configuration_revision=watch.active.configuration_revision,
                targets=watch.active.targets,
                task=watch.active.task,
                expires_at_ms=1,
            )
            watch.active = expired

            def persist_expired_lease(tx):
                snapshot = tx.get_mission(mission_id)
                updated = dict(snapshot)
                updated["watch_lease"] = asdict(expired)
                tx.update_mission(
                    updated,
                    expected_revision=int(snapshot["revision"]),
                    updated_at_ms=int(snapshot["updated_at_ms"]) + 1,
                )

            store.transact(persist_expired_lease)
            current = store.get_mission(mission_id)
            applied_count = len(watch.applied)
            rejected = await runtime.handle(
                _command(
                    "update_brief",
                    "expired-task-lease",
                    mission_id=mission_id,
                    revision=current["revision"],
                    args={"watch_task": "segment"},
                ),
                Principal("operator"),
                SCOPES,
                integrated=True,
            )
            assert not rejected["ok"]
            assert rejected["error"]["code"] == "watch_lease_expired"
            assert len(watch.applied) == applied_count
            assert watch.active is expired
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
