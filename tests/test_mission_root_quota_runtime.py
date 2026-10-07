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


def test_watch_root_full_pauses_and_explicit_resume_analyzes_fresh_frame(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("quota-watch-source", "epoch-1")
    jpeg = _jpeg()
    orphan = b"legacy orphan bytes" * 4
    root_limit = 2 * len(jpeg) + len(orphan) - 1
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
            assert orphan_path.read_bytes() == orphan
            media = tuple(evidence_root.glob("*.jpg"))
            assert len(media) == 1 and media[0].stat().st_size == len(jpeg)
            assert sum(path.stat().st_size for path in evidence_root.iterdir() if path.is_file()) + len(jpeg) > root_limit
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
        root_quota_bytes=2 * len(jpeg) + len(orphan) - 1,
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
