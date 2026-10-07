"""CPU runtime coverage for the read-only released-record API."""

import asyncio
import base64
import hashlib
import threading
import time
from io import BytesIO

import pytest
from PIL import Image

from visionbrain.mission_contracts import (
    Decision,
    GeometryItem,
    Principal,
    SourceBinding,
    SourceFrame,
    ToolResult,
    WatchLease,
    WatchLeaseRelease,
)
from visionbrain.mission_record_projection import MissionRecordProjectionError
from visionbrain.mission_records import (
    EvidenceRecord,
    Observation,
    deserialize_record,
    serialize_record,
    validate_record_bundle,
)
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (24, 18), (25, 110, 180)).save(output, format="JPEG")
    return output.getvalue()


class _Planner:
    def __init__(self, *, stop_watch=False):
        self.stop_watch = stop_watch

    def available(self, model_key):
        return model_key == "gemma"

    def plan(self, context):
        if not context.tool_results:
            return Decision(1, "detect_objects", {"targets": ["container"]})
        if self.stop_watch:
            return Decision(1, "finish")
        return Decision(1, "finish")

    def close(self):
        pass


class _Tools:
    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, request, _context):
        assert request.tool == "detect_objects"
        return ToolResult(
            "ok",
            items=(GeometryItem("item-1", "container", 0.9, (0.1, 0.2, 0.7, 0.8)),),
        )

    def close(self):
        pass


class _Watch:
    def __init__(self):
        self.active = None

    def current_configuration_revision(self):
        return 0 if self.active is None else self.active.configuration_revision

    def apply(self, request):
        revision = request.expected_configuration_revision + 1 if self.active is None else self.active.configuration_revision
        lease_id = "record-api-watch" if self.active is None else self.active.lease_id
        self.active = WatchLease(
            lease_id,
            request.mission_id,
            request.source_binding,
            revision,
            request.targets,
            request.task,
            request.expires_at_ms,
        )
        return self.active

    def release(self, lease, _reason):
        if self.active is not None and self.active.lease_id == lease.lease_id:
            self.active = None
        return WatchLeaseRelease(True, lease.configuration_revision + 1)


class _FrameSource:
    def __init__(self, binding):
        self.binding = binding

    def __call__(self, requested):
        if requested != self.binding:
            return None
        return SourceFrame(
            _jpeg(), self.binding.source_id, self.binding.source_epoch, 41, time.monotonic()
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
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = predicate()
        if result:
            return result
        await asyncio.sleep(0.005)
    raise AssertionError("runtime did not reach expected state")


def _roundtrip(records):
    round_trip = tuple(deserialize_record(serialize_record(record)) for record in records)
    assert round_trip == records
    validate_record_bundle(round_trip)
    return round_trip


def test_completed_inspect_uses_one_off_loop_store_read_and_preserves_projection_errors(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            _Planner(),
            _Tools(),
            lambda _binding: None,
            _Watch(),
            lambda _event: None,
            qualified_models={"gemma": ["inspect"]},
        )
        try:
            created = await runtime.handle(
                _command("create", "inspect-create", args={
                    "profile": {"id": "visual_inspection", "version": 1},
                    "expertise": "site inspector",
                    "mode": "inspect",
                    "reasoning_model": "gemma",
                }),
                Principal("installation"),
                SCOPES,
            )
            assert created["ok"], created
            initial = created["result"]["snapshot"]
            jpeg = _jpeg()
            attached = await runtime.handle(
                _command("attach_evidence", "inspect-evidence", initial["mission_id"], initial["revision"], {
                    "jpeg_b64": base64.b64encode(jpeg).decode(),
                    "sha256": hashlib.sha256(jpeg).hexdigest(),
                }),
                Principal("installation"),
                SCOPES,
            )
            snapshot = attached["result"]["snapshot"]
            await runtime.handle(
                _command("resume", "inspect-resume", snapshot["mission_id"], snapshot["revision"]),
                Principal("installation"),
                SCOPES,
            )
            await _wait_for(lambda: (
                current if (current := store.get_mission(snapshot["mission_id"]))["state"] == "completed" else None
            ))

            original_read = store.read_mission_record_rows
            read_threads = []

            def counted_read(mission_id):
                read_threads.append(threading.get_ident())
                return original_read(mission_id)

            store.read_mission_record_rows = counted_read
            event_loop_thread = threading.get_ident()
            records = await runtime.get_record_bundle(snapshot["mission_id"])
            assert len(read_threads) == 1
            assert read_threads[0] != event_loop_thread
            assert any(isinstance(record, Observation) for record in records)
            evidence_record = next(record for record in records if isinstance(record, EvidenceRecord))
            media_path = store.evidence_root / f"{evidence_record.evidence.evidence_id}.jpg"
            media_path.unlink()
            assert evidence_record.availability == "available"
            assert _roundtrip(records) == records

            def incomplete_read(mission_id):
                bundle = original_read(mission_id)
                bundle["evidence_rows"][0]["capture_time_ms"] = 1_800_000_000_000
                return bundle

            store.read_mission_record_rows = incomplete_read
            with pytest.raises(MissionRecordProjectionError, match="capture_time_provenance_missing"):
                await runtime.get_record_bundle(snapshot["mission_id"])
            store.read_mission_record_rows = original_read
            with pytest.raises(KeyError, match="mission-absent"):
                await runtime.get_record_bundle("mission-absent")

            original_project = runtime_module.project_mission_records
            projection_entered = threading.Event()
            projection_release = threading.Event()
            projection_threads = []

            def blocked_project(*args):
                projection_threads.append(threading.get_ident())
                projection_entered.set()
                if not projection_release.wait(timeout=1):
                    raise AssertionError("projection barrier was not released")
                return original_project(*args)

            monkeypatch.setattr(runtime_module, "project_mission_records", blocked_project)
            loop_thread = threading.get_ident()
            projection_task = asyncio.create_task(runtime.get_record_bundle(snapshot["mission_id"]))
            fail_safe = threading.Timer(0.2, projection_release.set)
            fail_safe.daemon = True
            fail_safe.start()
            try:
                assert await asyncio.wait_for(asyncio.to_thread(projection_entered.wait), timeout=1)
                await asyncio.sleep(0.01)
                assert not projection_task.done(), "projection blocked the event loop until its barrier opened"
                assert projection_threads and projection_threads[0] != loop_thread
            finally:
                projection_release.set()
                fail_safe.cancel()
            projected_again = await asyncio.wait_for(projection_task, timeout=1)
            assert _roundtrip(projected_again) == projected_again
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_completed_watch_bundle_roundtrips_source_and_frame_lineage(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("scout-record-api", "epoch-41")

    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            _Planner(stop_watch=True),
            _Tools(),
            _FrameSource(binding),
            _Watch(),
            lambda _event: None,
            qualified_models={"gemma": ["watch"]},
        )
        try:
            created = await runtime.handle(
                _command("create", "watch-create", args={
                    "profile": {"id": "visual_inspection", "version": 1},
                    "expertise": "site inspector",
                    "mode": "watch",
                    "reasoning_model": "gemma",
                    "source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch},
                }),
                Principal("installation"),
                SCOPES,
            )
            assert created["ok"], created
            initial = created["result"]["snapshot"]
            await runtime.handle(
                _command("resume", "watch-resume", initial["mission_id"], initial["revision"], {
                    "source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch},
                }),
                Principal("installation"),
                SCOPES,
            )
            paused = await _wait_for(lambda: (
                snapshot if (snapshot := store.get_mission(initial["mission_id"]))["state"] == "paused" and snapshot["cycle_history"] else None
            ))
            assert paused["reason"] == "watch_lease_missing"
            records = await runtime.get_record_bundle(initial["mission_id"])
            observation = next(record for record in records if isinstance(record, Observation))
            assert (observation.source_id, observation.source_epoch, observation.frame_id) == (
                binding.source_id, binding.source_epoch, 41
            )
            assert _roundtrip(records) == records
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
