"""Actual runtime save-boundary behavior when root evidence accounting blocks writes."""

import asyncio
import base64
import hashlib
import time
from io import BytesIO

from PIL import Image

from visionbrain.mission_contracts import (
    Decision,
    EvidenceArtifact,
    Principal,
    SourceBinding,
    SourceFrame,
    ToolResult,
    WatchLease,
    WatchLeaseRelease,
    WatchProposal,
)
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore, QuotaAccountingIncomplete


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg(color=(12, 90, 175)):
    output = BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="JPEG")
    return output.getvalue()


class _Planner:
    def __init__(self, *, watch=True):
        self.calls = 0
        self.watch = watch

    def available(self, model_key):
        return model_key == "gemma"

    def plan(self, context):
        self.calls += 1
        if not context.tool_results:
            return Decision(1, "detect_objects", {"targets": ["container"]})
        return Decision(1, "finish", watch=WatchProposal(("container",), "detect") if self.watch else None)

    def close(self):
        pass


class _Tools:
    def __init__(self, result=None):
        self.result = result or ToolResult("empty")
        self.calls = 0

    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, request, _context):
        self.calls += 1
        assert request.tool == "detect_objects"
        return self.result


class _Watch:
    def __init__(self):
        self.active = None
        self.applied = []
        self.released = []

    def current_configuration_revision(self):
        return self.active.configuration_revision if self.active else 0

    def apply(self, request):
        lease = WatchLease(
            "runtime-adaptive-lease",
            request.mission_id,
            request.source_binding,
            request.expected_configuration_revision + 1,
            request.targets,
            request.task,
            request.expires_at_ms,
        )
        self.active = lease
        self.applied.append(lease)
        return lease

    def manual_takeover(self, mission_id, binding):
        self.active = WatchLease(
            "operator-owned-lease",
            mission_id,
            binding,
            self.active.configuration_revision + 1,
            ("operator target",),
            "detect",
            int(time.time() * 1000) + 60_000,
        )

    def release(self, lease, reason):
        self.released.append((lease.lease_id, reason))
        if (
            self.active is not None
            and self.active.lease_id == lease.lease_id
            and self.active.configuration_revision == lease.configuration_revision
        ):
            self.active = None
            return WatchLeaseRelease(True, lease.configuration_revision + 1)
        return WatchLeaseRelease(False, self.current_configuration_revision())


class _LatestFrame:
    def __init__(self, binding, jpeg, watch, *, manual_takeover=False):
        self.binding = binding
        self.jpeg = jpeg
        self.watch = watch
        self.manual_takeover = manual_takeover
        self.calls = 0
        self.manual_takeover_applied = False

    def __call__(self, requested):
        assert requested == self.binding
        self.calls += 1
        if self.calls <= 2:
            frame_id = 1
        elif self.calls == 3:
            frame_id = 2
            if self.manual_takeover and not self.manual_takeover_applied:
                self.watch.manual_takeover("operator-session", self.binding)
                self.manual_takeover_applied = True
        else:
            # A distinct latest frame for explicit resume, then held stable so
            # the test does not create a second post-resume quota failure.
            frame_id = 3
        return SourceFrame(
            self.jpeg,
            self.binding.source_id,
            self.binding.source_epoch,
            frame_id,
            time.monotonic(),
        )


def _command(name, request_id, mission_id=None, revision=None, args=None):
    message = {
        "type": "mission_command",
        "schema_version": 1,
        "request_id": request_id,
        "command": name,
        "mission_id": mission_id,
        "args": args or {},
    }
    if revision is not None:
        message["expected_revision"] = revision
    return message


def _attachment(jpeg, *, closeup_request_id=None):
    args = {
        "jpeg_b64": base64.b64encode(jpeg).decode(),
        "sha256": hashlib.sha256(jpeg).hexdigest(),
    }
    if closeup_request_id is not None:
        args["closeup_request_id"] = closeup_request_id
    return args


async def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before bounded timeout")


async def _create(runtime, *, mode, binding=None, request_id="quota-create"):
    return await runtime.handle(
        _command("create", request_id, args={
            "profile": {"id": "visual_inspection", "version": 1},
            "expertise": "site inspector",
            "mode": mode,
            "reasoning_model": "gemma",
            **({"source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch}} if binding else {}),
        }),
        Principal("installation"),
        SCOPES,
    )


def _runtime(store, planner, tools, source, watch):
    return MissionRuntime(
        store,
        planner,
        tools,
        source,
        watch,
        lambda _event: None,
        qualified_models={"gemma": ["inspect", "watch"]},
    )


def _stored_origin(store, evidence_id):
    with store._lock:
        row = store._connection.execute(
            "SELECT origin FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
    return row[0] if row else None


def test_watch_root_full_pauses_and_explicit_resume_analyzes_fresh_frame(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("quota-watch-source", "epoch-1")
    jpeg = _jpeg()
    orphan = b"legacy orphan bytes" * 4
    root_limit = 2 * len(jpeg) + len(orphan)
    evidence_root = tmp_path / "evidence"
    store = MissionStore(tmp_path / "missions.sqlite3", evidence_root, root_quota_bytes=root_limit)
    orphan_path = evidence_root / "unindexed-orphan.bin"
    orphan_path.write_bytes(orphan)
    watch = _Watch()
    source = _LatestFrame(binding, jpeg, watch)
    planner, tools = _Planner(), _Tools()
    runtime = _runtime(store, planner, tools, source, watch)

    async def scenario():
        try:
            created = await _create(runtime, mode="watch", binding=binding)
            assert created["ok"]
            initial = created["result"]["snapshot"]
            old_crop = store.transact(
                lambda tx: tx.save_evidence(
                    initial["mission_id"], jpeg, kind="crop", origin="generated_crop", created_at_ms=1
                )
            ).value
            old_crop_path = evidence_root / f"{old_crop['evidence_id']}.jpg"
            resumed = await runtime.handle(
                _command(
                    "resume", "quota-resume", initial["mission_id"], initial["revision"],
                    {"source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch}},
                ),
                Principal("installation"), SCOPES,
            )
            assert resumed["ok"]

            paused = await _wait_for(lambda: (
                snapshot
                if (snapshot := store.get_mission(initial["mission_id"]))["state"] == "paused"
                and snapshot["reason"] == "evidence_quota_full"
                and watch.released
                else None
            ))
            assert paused["watch_lease"] is None
            assert watch.applied
            assert watch.released == [(watch.applied[0].lease_id, "evidence_quota_full")]
            assert watch.active is None
            saved_watch_frame = next(
                ref for ref in paused["evidence"]
                if ref.get("kind") == "frame" and ref.get("available", True)
            )
            assert _stored_origin(store, saved_watch_frame["evidence_id"]) == "watch_frame"
            assert orphan_path.read_bytes() == orphan
            assert not old_crop_path.exists()
            assert any(
                ref.get("evidence_id") == old_crop["evidence_id"]
                and ref.get("availability_reason") == "rolled_off"
                for ref in paused["evidence"]
            )
            media = tuple(evidence_root.glob("*.jpg"))
            assert len(media) == 1 and media[0].stat().st_size == len(jpeg)
            assert sum(path.stat().st_size for path in evidence_root.iterdir() if path.is_file()) + len(jpeg) == root_limit
            assert source.calls >= 3 and tools.calls == 1

            # Correcting space does not schedule a background retry.
            orphan_path.unlink()
            calls_at_pause = source.calls
            plans_at_pause = planner.calls
            await asyncio.sleep(0.04)
            assert store.get_mission(initial["mission_id"])["state"] == "paused"
            assert source.calls == calls_at_pause
            assert planner.calls == plans_at_pause

            explicit = await runtime.handle(
                _command(
                    "resume", "quota-explicit-resume", initial["mission_id"], paused["revision"],
                    {"source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch}},
                ),
                Principal("installation"), SCOPES,
            )
            assert explicit["ok"], explicit
            await _wait_for(lambda: tools.calls >= 2)
            analyzed = store.get_mission(initial["mission_id"])
            assert analyzed["state"] == "running"
            assert any(item.get("frame_id") == 3 and item.get("available", True) for item in analyzed["evidence"])
            assert tools.calls == 2
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_watch_quota_pause_compare_releases_only_its_lease_after_manual_takeover(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("quota-manual-source", "epoch-1")
    jpeg = _jpeg()
    orphan = b"legacy orphan bytes" * 4
    evidence_root = tmp_path / "evidence"
    store = MissionStore(
        tmp_path / "missions.sqlite3", evidence_root,
        root_quota_bytes=2 * len(jpeg) + len(orphan) - 1,
    )
    (evidence_root / "unindexed-orphan.bin").write_bytes(orphan)
    watch = _Watch()
    source = _LatestFrame(binding, jpeg, watch, manual_takeover=True)
    runtime = _runtime(store, _Planner(), _Tools(), source, watch)

    async def scenario():
        try:
            created = await _create(runtime, mode="watch", binding=binding, request_id="manual-create")
            snapshot = created["result"]["snapshot"]
            resumed = await runtime.handle(
                _command("resume", "manual-resume", snapshot["mission_id"], snapshot["revision"],
                         {"source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch}}),
                Principal("installation"), SCOPES,
            )
            assert resumed["ok"]
            paused = await _wait_for(lambda: (
                current
                if (current := store.get_mission(snapshot["mission_id"]))["state"] == "paused"
                and current["reason"] == "evidence_quota_full"
                and watch.released
                else None
            ))
            assert watch.released == [(watch.applied[0].lease_id, "evidence_quota_full")]
            assert paused["watch_lease"] is None
            assert watch.active is not None and watch.active.lease_id == "operator-owned-lease"
            assert watch.active.targets == ("operator target",)
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_attach_root_full_pauses_and_idempotent_failure_does_not_retry_after_space_correction(tmp_path):
    jpeg = _jpeg()
    orphan = b"legacy orphan bytes" * 4
    evidence_root = tmp_path / "evidence"
    store = MissionStore(
        tmp_path / "missions.sqlite3", evidence_root,
        root_quota_bytes=len(jpeg) + len(orphan) - 1,
    )
    orphan_path = evidence_root / "unindexed-orphan.bin"
    orphan_path.write_bytes(orphan)
    runtime = _runtime(store, _Planner(watch=False), _Tools(), lambda _binding: None, _Watch())

    async def scenario():
        try:
            created = await _create(runtime, mode="inspect", request_id="attach-create")
            snapshot = created["result"]["snapshot"]
            attachment = _command(
                "attach_evidence", "attach-root-full", snapshot["mission_id"], snapshot["revision"],
                {"jpeg_b64": base64.b64encode(jpeg).decode(), "sha256": hashlib.sha256(jpeg).hexdigest()},
            )
            rejected = await runtime.handle(attachment, Principal("installation"), SCOPES)
            assert not rejected["ok"]
            assert rejected["error"]["code"] == "evidence_quota_full"
            paused = rejected["result"]["snapshot"]
            assert paused["state"] == "paused" and paused["reason"] == "evidence_quota_full"
            assert paused["watch_lease"] is None
            assert "resume explicitly" in paused["activity"]
            assert store.get_mission(snapshot["mission_id"])["evidence"] == []

            orphan_path.unlink()
            replay = await runtime.handle(attachment, Principal("installation"), SCOPES)
            assert replay == rejected
            assert store.get_mission(snapshot["mission_id"])["evidence"] == []
            assert store.get_mission(snapshot["mission_id"])["state"] == "paused"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_crop_root_full_pauses_inspect_and_records_rejected_artifact(tmp_path):
    jpeg = _jpeg()
    orphan = b"legacy orphan bytes" * 4
    evidence_root = tmp_path / "evidence"
    store = MissionStore(
        tmp_path / "missions.sqlite3", evidence_root,
        root_quota_bytes=2 * len(jpeg) + len(orphan),
    )
    orphan_path = evidence_root / "unindexed-orphan.bin"
    orphan_path.write_bytes(orphan)
    planner = _Planner(watch=False)
    tools = _Tools()
    runtime = _runtime(store, planner, tools, lambda _binding: None, _Watch())

    async def scenario():
        try:
            created = await _create(runtime, mode="inspect", request_id="crop-create")
            snapshot = created["result"]["snapshot"]
            attached = await runtime.handle(
                _command(
                    "attach_evidence", "crop-attach", snapshot["mission_id"], snapshot["revision"],
                    {"jpeg_b64": base64.b64encode(jpeg).decode(), "sha256": hashlib.sha256(jpeg).hexdigest()},
                ),
                Principal("installation"), SCOPES,
            )
            assert attached["ok"], attached
            input_id = attached["result"]["evidence_id"]
            old_crop = store.transact(
                lambda tx: tx.save_evidence(
                    snapshot["mission_id"], jpeg, kind="crop", created_at_ms=1,
                    origin="generated_crop",
                    parent_evidence_id=input_id,
                )
            ).value
            old_crop_path = evidence_root / f"{old_crop['evidence_id']}.jpg"
            tools.result = ToolResult(
                "ok",
                artifacts=(EvidenceArtifact(jpeg, "crop", input_id, (0.1, 0.1, 0.8, 0.8)),),
            )
            current = attached["result"]["snapshot"]
            resumed = await runtime.handle(
                _command("resume", "crop-resume", snapshot["mission_id"], current["revision"]),
                Principal("installation"), SCOPES,
            )
            assert resumed["ok"], resumed
            paused = await _wait_for(lambda: (
                result
                if (result := store.get_mission(snapshot["mission_id"]))["state"] == "paused"
                and result["reason"] == "evidence_quota_full"
                else None
            ))
            assert paused["input_evidence_id"] == input_id
            assert not old_crop_path.exists()
            assert orphan_path.read_bytes() == orphan
            assert any(
                ref.get("evidence_id") == old_crop["evidence_id"]
                and ref.get("availability_reason") == "rolled_off"
                for ref in paused["evidence"]
            )
            assert tools.calls == 1
            records = store.tool_records(snapshot["mission_id"])
            assert len(records) == 1
            assert records[0]["error_code"] == "evidence_quota_full"
            assert records[0]["evidence_ids"] == []
            assert orphan_path.read_bytes() == orphan
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_attach_root_rolloff_protects_references_and_replays_committed_refusal(tmp_path, monkeypatch):
    from visionbrain import mission_store as store_module

    jpeg = _jpeg()
    other_frame = _jpeg((30, 120, 185))
    incoming = _jpeg((190, 80, 25))
    evidence_root = tmp_path / "evidence"
    store = MissionStore(tmp_path / "missions.sqlite3", evidence_root, root_quota_bytes=10_000_000)
    runtime = _runtime(store, _Planner(watch=False), _Tools(), lambda _binding: None, _Watch())
    cleanup_calls = []
    original_cleanup = store_module._cleanup_retired_paths

    def observe_cleanup(paths):
        if paths:
            cleanup_calls.append(tuple(paths))
            saved = store.get_mission(mission_id)
            assert saved["state"] == "waiting_evidence"
            assert saved["reason"] == "evidence_quota_full"
            assert any(
                ref.get("evidence_id") == oldest_eligible["evidence_id"]
                and ref.get("availability_reason") == "rolled_off"
                for ref in saved["evidence"]
            )
            assert (evidence_root / f"{oldest_eligible['evidence_id']}.jpg").exists()
        return original_cleanup(paths)

    mission_id = None
    oldest_eligible = None
    monkeypatch.setattr(store_module, "_cleanup_retired_paths", observe_cleanup)

    async def scenario():
        nonlocal mission_id, oldest_eligible
        try:
            created = await _create(runtime, mode="inspect", request_id="protected-create")
            snapshot = created["result"]["snapshot"]
            mission_id = snapshot["mission_id"]
            first = await runtime.handle(
                _command("attach_evidence", "explicit-frame-one", mission_id, snapshot["revision"], _attachment(jpeg)),
                Principal("installation", evidence_kind="frame", source_id="camera", source_epoch="epoch-1", frame_id=1),
                SCOPES,
            )
            assert first["ok"], first
            second = await runtime.handle(
                _command("attach_evidence", "explicit-frame-two", mission_id, first["result"]["snapshot"]["revision"], _attachment(other_frame)),
                Principal("installation", evidence_kind="frame", source_id="camera", source_epoch="epoch-1", frame_id=2),
                SCOPES,
            )
            assert second["ok"], second
            snapshot = second["result"]["snapshot"]
            rows = {}

            def seed(tx):
                for key, kind, created_at in (
                    ("closeup", "closeup", 10),
                    ("accepted", "crop", 20),
                    ("rejected", "crop", 30),
                    ("observed", "crop", 40),
                    ("exported", "crop", 50),
                    ("oldest", "crop", 60),
                    ("newer", "crop", 70),
                ):
                    rows[key] = tx.save_evidence(
                        mission_id, jpeg, kind=kind,
                        origin="generated_crop" if kind == "crop" else "unknown",
                        created_at_ms=created_at,
                    )
                tx.pin_exported_evidence(mission_id, [rows["exported"]["evidence_id"]])
                current = tx.get_mission(mission_id)
                current["state"] = "waiting_evidence"
                current["reason"] = "closeup_requested"
                current["closeup_request"] = {
                    "request_id": "closeup-1",
                    "evidence_id": rows["closeup"]["evidence_id"],
                    "expires_at_ms": int(time.time() * 1000) + 60_000,
                }
                current["findings"] = [
                    {"finding_id": "accepted-finding", "evidence_id": rows["accepted"]["evidence_id"], "review": {"decision": "accepted"}},
                    {"finding_id": "rejected-finding", "evidence_id": rows["rejected"]["evidence_id"], "review": {"decision": "rejected"}},
                    {"finding_id": "observed-finding", "observations": [{"evidence_id": rows["observed"]["evidence_id"]}]},
                ]
                current["evidence"] = list(current.get("evidence", ())) + list(rows.values())
                return tx.update_mission(
                    current, expected_revision=int(current["revision"]), updated_at_ms=int(time.time() * 1000)
                )

            snapshot = store.transact(seed).value
            mission_id = snapshot["mission_id"]
            oldest_eligible = rows["oldest"]
            protected_ids = {
                first["result"]["evidence_id"],
                second["result"]["evidence_id"],
                rows["closeup"]["evidence_id"],
                rows["accepted"]["evidence_id"],
                rows["rejected"]["evidence_id"],
                rows["observed"]["evidence_id"],
                rows["exported"]["evidence_id"],
            }
            protected_paths = {evidence_root / f"{evidence_id}.jpg" for evidence_id in protected_ids}
            oldest_path = evidence_root / f"{oldest_eligible['evidence_id']}.jpg"
            newer_path = evidence_root / f"{rows['newer']['evidence_id']}.jpg"
            initial_root_usage = store._root_evidence_usage()
            store.root_quota_bytes = initial_root_usage + len(incoming) - len(jpeg)

            invalid = await runtime.handle(
                _command("attach_evidence", "attach-unauthorized", mission_id, snapshot["revision"], _attachment(incoming)),
                Principal("observer"), {"mission:read"},
            )
            stale = await runtime.handle(
                _command("attach_evidence", "attach-stale", mission_id, snapshot["revision"] - 1,
                         _attachment(incoming, closeup_request_id="closeup-1")),
                Principal("installation"), SCOPES,
            )
            wrong_closeup = await runtime.handle(
                _command("attach_evidence", "attach-wrong-closeup", mission_id, snapshot["revision"],
                         _attachment(incoming, closeup_request_id="wrong-closeup")),
                Principal("installation"), SCOPES,
            )
            assert invalid["error"]["code"] == "unauthorized"
            assert stale["error"]["code"] == "revision_conflict"
            assert wrong_closeup["error"]["code"] == "closeup_mismatch"
            assert cleanup_calls == []
            assert all(path.exists() for path in protected_paths | {oldest_path, newer_path})
            assert store.get_mission(mission_id) == snapshot

            original = _command(
                "attach_evidence", "attach-root-rolloff", mission_id, snapshot["revision"],
                _attachment(incoming, closeup_request_id="closeup-1"),
            )
            refused = await runtime.handle(original, Principal("installation"), SCOPES)
            assert not refused["ok"]
            assert refused["error"]["code"] == "evidence_quota_full"
            assert refused["result"]["execution_outcome"] == "waiting_evidence"
            assert refused["result"]["snapshot"]["closeup_request"] == snapshot["closeup_request"]
            assert cleanup_calls == [(oldest_path,)]
            assert not oldest_path.exists()
            assert newer_path.exists()
            assert all(path.exists() for path in protected_paths)

            replay = await runtime.handle(original, Principal("installation"), SCOPES)
            assert replay == refused
            conflict = dict(original)
            conflict["args"] = _attachment(_jpeg((5, 5, 5)), closeup_request_id="closeup-1")
            conflicting_replay = await runtime.handle(conflict, Principal("installation"), SCOPES)
            assert conflicting_replay["error"]["code"] == "invalid_request"
            assert cleanup_calls == [(oldest_path,)]
            assert store.get_mission(mission_id)["state"] == "waiting_evidence"

            updated = refused["result"]["snapshot"]
            retry = await runtime.handle(
                _command("attach_evidence", "attach-root-rolloff-retry", mission_id, updated["revision"],
                         _attachment(incoming, closeup_request_id="closeup-1")),
                Principal("installation"), SCOPES,
            )
            assert retry["ok"], retry
            assert retry["result"]["snapshot"]["closeup_request"] is None
            assert replay == refused
            assert len(cleanup_calls) == 1
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_root_rolloff_retry_rescans_after_unlink_failure_or_competing_writer(tmp_path, monkeypatch):
    from pathlib import Path
    from visionbrain import mission_store as store_module

    async def run_case(case):
        jpeg = _jpeg()
        evidence_root = tmp_path / case / "evidence"
        store = MissionStore(tmp_path / case / "missions.sqlite3", evidence_root, root_quota_bytes=10_000_000)
        runtime = _runtime(store, _Planner(watch=False), _Tools(), lambda _binding: None, _Watch())
        other_store = None
        cleanup_calls = []
        original_cleanup = store_module._cleanup_retired_paths

        def observe_cleanup(paths):
            if paths:
                cleanup_calls.extend(paths)
            return original_cleanup(paths)

        async def scenario():
            nonlocal other_store
            try:
                monkeypatch.setattr(store_module, "_cleanup_retired_paths", observe_cleanup)
                created = await _create(runtime, mode="inspect", request_id=f"{case}-create")
                snapshot = created["result"]["snapshot"]
                attached = await runtime.handle(
                    _command("attach_evidence", f"{case}-input", snapshot["mission_id"], snapshot["revision"], _attachment(jpeg)),
                    Principal("installation", evidence_kind="frame", source_id="camera", source_epoch="epoch-1", frame_id=1),
                    SCOPES,
                )
                mission_id = snapshot["mission_id"]
                crop = store.transact(
                    lambda tx: tx.save_evidence(
                        mission_id, jpeg, kind="crop", origin="generated_crop", created_at_ms=1
                    )
                ).value
                crop_path = evidence_root / f"{crop['evidence_id']}.jpg"
                latest = attached["result"]["snapshot"]
                now = int(time.time() * 1000)
                current = dict(latest)
                current.update(
                    state="waiting_evidence",
                    reason="closeup_requested",
                    closeup_request={"request_id": "closeup-retry", "evidence_id": crop["evidence_id"], "expires_at_ms": now + 60_000},
                )
                latest = store.transact(
                    lambda tx: tx.update_mission(current, expected_revision=current["revision"], updated_at_ms=now)
                ).value
                # The close-up source is protected; use a second, unreferenced crop as the eligible row.
                eligible = store.transact(
                    lambda tx: tx.save_evidence(
                        mission_id, jpeg, kind="crop", origin="generated_crop", created_at_ms=2
                    )
                ).value
                eligible_path = evidence_root / f"{eligible['evidence_id']}.jpg"
                current = store.get_mission(mission_id)
                current["evidence"].extend([crop, eligible])
                latest = store.transact(
                    lambda tx: tx.update_mission(current, expected_revision=current["revision"], updated_at_ms=now + 1)
                ).value
                # Make the eligible crop older while keeping the close-up crop first but protected.
                store.root_quota_bytes = store._root_evidence_usage()

                if case == "unlink_failure":
                    original_unlink = Path.unlink

                    def fail_target(path, *args, **kwargs):
                        if path == eligible_path:
                            raise OSError("injected unlink failure")
                        return original_unlink(path, *args, **kwargs)

                    monkeypatch.setattr(Path, "unlink", fail_target)

                refusal = await runtime.handle(
                    _command("attach_evidence", f"{case}-first", mission_id, latest["revision"],
                             _attachment(jpeg, closeup_request_id="closeup-retry")),
                    Principal("installation"), SCOPES,
                )
                assert refusal["error"]["code"] == "evidence_quota_full"
                assert cleanup_calls == [eligible_path]
                assert crop_path.exists()
                assert store.get_mission(mission_id)["state"] == "waiting_evidence"
                assert (eligible_path.exists()) is (case == "unlink_failure")
                replay = await runtime.handle(
                    _command("attach_evidence", f"{case}-first", mission_id, latest["revision"],
                             _attachment(jpeg, closeup_request_id="closeup-retry")),
                    Principal("installation"), SCOPES,
                )
                assert replay == refusal
                assert cleanup_calls == [eligible_path]

                if case == "competing_writer":
                    other_store = MissionStore(
                        tmp_path / case / "other.sqlite3", evidence_root,
                        root_quota_bytes=store.root_quota_bytes,
                    )
                    # The neighboring database's own mission is not a candidate for this runtime.
                    other_store.transact(lambda tx: tx.insert_mission({
                        "mission_id": "other-mission", "revision": 1, "state": "ready", "updated_at_ms": now,
                        "evidence": [], "findings": [],
                    }))
                    competing = other_store.transact(
                        lambda tx: tx.save_evidence("other-mission", jpeg, kind="imported", created_at_ms=now)
                    ).value
                    competing_path = evidence_root / f"{competing['evidence_id']}.jpg"
                    assert competing_path.exists()

                # A fresh request must scan physical bytes and refuse without another eligible row.
                paused = store.get_mission(mission_id)
                retry = await runtime.handle(
                    _command("attach_evidence", f"{case}-second", mission_id, paused["revision"],
                             _attachment(jpeg, closeup_request_id="closeup-retry")),
                    Principal("installation"), SCOPES,
                )
                assert retry["error"]["code"] == "evidence_quota_full"
                assert cleanup_calls == [eligible_path]
                assert store.get_mission(mission_id)["state"] == "waiting_evidence"
                assert store.get_mission(mission_id)["reason"] == "evidence_quota_full"
                assert crop_path.exists()
                if case == "unlink_failure":
                    assert eligible_path.exists()
                else:
                    assert not eligible_path.exists()
                    assert competing_path.exists()
            finally:
                await runtime.close()
                if other_store is not None:
                    other_store.close()
                store.close()

        await scenario()

    asyncio.run(run_case("unlink_failure"))
    monkeypatch.undo()
    asyncio.run(run_case("competing_writer"))


def test_explicit_unknown_and_watch_frame_origins_gate_both_rolloff_paths(tmp_path):
    jpeg = _jpeg()

    async def run_case(quota_kind):
        evidence_root = tmp_path / quota_kind / "evidence"
        store = MissionStore(
            tmp_path / quota_kind / "missions.sqlite3",
            evidence_root,
            quota_bytes=1_000_000,
            root_quota_bytes=1_000_000,
        )
        runtime = _runtime(store, _Planner(watch=False), _Tools(), lambda _binding: None, _Watch())
        try:
            created = await _create(runtime, mode="inspect", request_id=f"origin-{quota_kind}-create")
            snapshot = created["result"]["snapshot"]
            attached = await runtime.handle(
                _command("attach_evidence", f"origin-{quota_kind}-input", snapshot["mission_id"], snapshot["revision"], _attachment(jpeg)),
                Principal("installation", evidence_kind="frame", source_id="camera", source_epoch="epoch-1", frame_id=1),
                SCOPES,
            )
            assert attached["ok"], attached
            mission_id = snapshot["mission_id"]
            attached_id = attached["result"]["evidence_id"]
            assert _stored_origin(store, attached_id) == "explicit_attachment"

            def seed(tx):
                rows = [
                    tx.save_evidence(mission_id, jpeg, kind="frame", origin="explicit_attachment", created_at_ms=1),
                    tx.save_evidence(mission_id, jpeg, kind="frame", origin="unknown", created_at_ms=2),
                    tx.save_evidence(mission_id, jpeg, kind="frame", origin="watch_frame", created_at_ms=3),
                ]
                current = tx.get_mission(mission_id)
                current["evidence"].extend(rows)
                if quota_kind == "count":
                    from visionbrain.mission_contracts import MAX_MISSION_EVIDENCE

                    count, _bytes = tx.evidence_usage(mission_id)
                    for index in range(MAX_MISSION_EVIDENCE - count):
                        filler = tx.save_evidence(
                            mission_id, jpeg, kind="imported", origin="unknown",
                            created_at_ms=10 + index,
                        )
                        current["evidence"].append(filler)
                saved = tx.update_mission(
                    current, expected_revision=current["revision"], updated_at_ms=int(time.time() * 1000)
                )
                return rows, saved

            rows, snapshot = store.transact(seed).value
            explicit, unknown, automatic = rows
            paths = {
                "attached": evidence_root / f"{attached_id}.jpg",
                "explicit": evidence_root / f"{explicit['evidence_id']}.jpg",
                "unknown": evidence_root / f"{unknown['evidence_id']}.jpg",
                "automatic": evidence_root / f"{automatic['evidence_id']}.jpg",
            }

            caller_origin = await runtime.handle(
                _command(
                    "attach_evidence", f"origin-{quota_kind}-caller-origin", mission_id,
                    snapshot["revision"], _attachment(jpeg) | {"origin": "watch_frame"},
                ),
                Principal("installation"), SCOPES,
            )
            assert caller_origin["error"]["code"] == "invalid_request"
            assert store.get_mission(mission_id) == snapshot

            usage = store._root_evidence_usage()
            if quota_kind == "root":
                store.root_quota_bytes = usage
            elif quota_kind == "mission":
                store.quota_bytes = store.transact(lambda tx: tx.total_evidence_usage()[1]).value

            result = await runtime.handle(
                _command(
                    "attach_evidence", f"origin-{quota_kind}-rolloff", mission_id,
                    snapshot["revision"], _attachment(jpeg),
                ),
                Principal("installation", evidence_kind="frame", source_id="camera", source_epoch="epoch-1", frame_id=2),
                SCOPES,
            )
            if quota_kind == "root":
                assert result["error"]["code"] == "evidence_quota_full"
                saved = result["result"]["snapshot"]
            else:
                assert result["ok"], result
                saved = result["result"]["snapshot"]
                new_id = result["result"]["evidence_id"]
                assert _stored_origin(store, new_id) == "explicit_attachment"

            assert paths["attached"].exists()
            assert paths["explicit"].exists()
            assert paths["unknown"].exists()
            assert not paths["automatic"].exists()
            assert any(
                ref.get("evidence_id") == automatic["evidence_id"]
                and ref.get("availability_reason") == "rolled_off"
                for ref in saved["evidence"]
            )
            assert all("origin" not in ref for ref in saved["evidence"])
        finally:
            await runtime.close()
            store.close()

    asyncio.run(run_case("root"))
    asyncio.run(run_case("mission"))
    asyncio.run(run_case("count"))


def test_watch_accounting_incomplete_pauses_with_distinct_operator_reason(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("quota-unknown-source", "epoch-1")
    jpeg = _jpeg()
    evidence_root = tmp_path / "evidence"
    store = MissionStore(tmp_path / "missions.sqlite3", evidence_root, root_quota_bytes=1_000_000)
    original_usage = store._root_evidence_usage
    scans = 0

    def incomplete_after_first_saved_frame():
        nonlocal scans
        scans += 1
        if scans == 2:
            raise QuotaAccountingIncomplete("injected bounded root-lock contention")
        return original_usage()

    monkeypatch.setattr(store, "_root_evidence_usage", incomplete_after_first_saved_frame)
    watch = _Watch()
    runtime = _runtime(store, _Planner(), _Tools(), _LatestFrame(binding, jpeg, watch), watch)

    async def scenario():
        try:
            created = await _create(runtime, mode="watch", binding=binding, request_id="unknown-create")
            snapshot = created["result"]["snapshot"]
            resumed = await runtime.handle(
                _command("resume", "unknown-resume", snapshot["mission_id"], snapshot["revision"],
                         {"source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch}}),
                Principal("installation"), SCOPES,
            )
            assert resumed["ok"]
            paused = await _wait_for(lambda: (
                result
                if (result := store.get_mission(snapshot["mission_id"]))["state"] == "paused"
                and result["reason"] == "evidence_quota_accounting_unavailable"
                and watch.released
                else None
            ))
            assert paused["watch_lease"] is None
            assert "accounted safely" in paused["activity"]
            assert "resume explicitly" in paused["activity"]
            assert watch.released == [(watch.applied[0].lease_id, "evidence_quota_accounting_unavailable")]
            assert len(tuple(evidence_root.glob("*.jpg"))) == 1
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
