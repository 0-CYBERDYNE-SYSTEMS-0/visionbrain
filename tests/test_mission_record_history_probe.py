"""CPU probe for Watch record history across a pause/resume generation change."""

import asyncio
import time
from io import BytesIO

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
from visionbrain.mission_records import Finding, Observation
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (24, 18), (25, 110, 180)).save(output, format="JPEG")
    return output.getvalue()


class _Planner:
    def available(self, model_key):
        return model_key == "gemma"

    def plan(self, context):
        if not context.tool_results:
            return Decision(1, "detect_objects", {"targets": ["container"]})
        tool = context.tool_results[-1]
        return Decision(
            1,
            "finish",
            findings=(FindingProposal(
                "Container is present.", "localized_object", (),
                item_refs=((tool.tool_result_id, "container-1"),),
            ),),
            watch=WatchProposal(("container",), "detect"),
        )

    def close(self):
        pass


class _Tools:
    def __init__(self, frames):
        self.frames = frames

    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, request, _context):
        assert request.tool == "detect_objects"
        self.frames.frame_id += 1
        return ToolResult(
            "ok",
            items=(GeometryItem("container-1", "container", 0.9, (0.1, 0.2, 0.7, 0.8)),),
        )

    def close(self):
        pass


class _Watch:
    def __init__(self):
        self.active = None
        self.revision = 0

    def current_configuration_revision(self):
        return self.revision if self.active is None else self.active.configuration_revision

    def apply(self, request):
        self.revision = request.expected_configuration_revision + 1
        self.active = WatchLease(
            "history-probe", request.mission_id, request.source_binding,
            self.revision, request.targets,
            request.task, request.expires_at_ms,
        )
        return self.active

    def release(self, lease, _reason):
        self.active = None
        self.revision = lease.configuration_revision + 1
        return WatchLeaseRelease(True, self.revision)


class _Frames:
    def __init__(self, binding):
        self.binding = binding
        self.frame_id = 1

    def __call__(self, requested):
        if requested != self.binding:
            return None
        return SourceFrame(
            _jpeg(), self.binding.source_id, self.binding.source_epoch,
            self.frame_id, time.monotonic(),
        )


def _command(name, request_id, mission_id=None, revision=None, args=None):
    value = {
        "type": "mission_command", "schema_version": 1,
        "request_id": request_id, "command": name, "mission_id": mission_id,
        "args": args or {},
    }
    if revision is not None:
        value["expected_revision"] = revision
    return value


async def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := predicate():
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("Watch did not complete the expected cycle")


async def _resume_after_drain(runtime, mission_id, revision, binding):
    deadline = time.monotonic() + 3.0
    attempt = 0
    while time.monotonic() < deadline:
        reply = await runtime.handle(_command("resume", f"resume-2-{attempt}", mission_id,
            revision, {"source_binding": {
                "source_id": binding.source_id, "source_epoch": binding.source_epoch,
            }}), Principal("installation"), SCOPES)
        if (reply.get("error") or {}).get("code") != "runtime_busy":
            return reply
        attempt += 1
        await asyncio.sleep(0.005)
    raise AssertionError("previous Watch generation did not drain")


def test_watch_record_bundle_preserves_completed_history_after_pause_resume(tmp_path, monkeypatch):
    """Completed Watch cycles remain projectable after their generation becomes stale."""
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("history-probe", "epoch-1")
    frames = _Frames(binding)

    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store, _Planner(), _Tools(frames), frames, _Watch(), lambda _event: None,
            qualified_models={"gemma": ["watch"]},
        )
        try:
            created = await runtime.handle(_command("create", "create", args={
                "profile": {"id": "visual_inspection", "version": 1},
                "expertise": "site inspector", "mode": "watch", "reasoning_model": "gemma",
                "source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch},
            }), Principal("installation"), SCOPES)
            assert created["ok"], created
            mission_id = created["result"]["snapshot"]["mission_id"]
            first = await runtime.handle(_command("resume", "resume-1", mission_id,
                created["result"]["snapshot"]["revision"], {"source_binding": {
                    "source_id": binding.source_id, "source_epoch": binding.source_epoch,
                }}), Principal("installation"), SCOPES)
            assert first["ok"], first
            completed_first = await _wait_for(
                lambda: store.get_mission(mission_id) if len(store.get_mission(mission_id)["cycle_history"]) == 1 else None
            )
            paused = await runtime.handle(_command("pause", "pause-1", mission_id,
                completed_first["revision"]), Principal("installation"), SCOPES)
            assert paused["ok"], paused
            resumed = await _resume_after_drain(
                runtime, mission_id, paused["result"]["snapshot"]["revision"], binding,
            )
            assert resumed["ok"], resumed.get("error")
            completed_second = await _wait_for(
                lambda: store.get_mission(mission_id) if len(store.get_mission(mission_id)["cycle_history"]) == 2 else None
            )
            final_pause = await runtime.handle(_command("pause", "pause-2", mission_id,
                completed_second["revision"]), Principal("installation"), SCOPES)
            assert final_pause["ok"], final_pause
            paused_snapshot = final_pause["result"]["snapshot"]
            finding_id = paused_snapshot["findings"][0]["finding_id"]
            reviewed = await runtime.handle(_command("review_finding", "review-history", mission_id,
                paused_snapshot["revision"], {
                    "finding_id": finding_id, "decision": "accepted", "note": "Reviewed historical result.",
                }), Principal("operator-2"), SCOPES)
            assert reviewed["ok"], reviewed

            rows = store.read_mission_record_rows(mission_id)
            snapshot = rows["snapshot"]
            cycles = snapshot["cycle_history"]
            assert len(cycles) == 2
            assert cycles[0]["execution_generation"] < cycles[1]["execution_generation"]
            assert snapshot["state"] == "paused" and snapshot["cycle_id"] is None
            assert all(cycle["execution_generation"] < snapshot["execution_generation"] for cycle in cycles)
            # SQL is_current records commit validity; generation carries current authority.
            assert all(row["is_current"] for row in rows["tool_rows"])

            original = {
                row["cycle_id"]: (row["record"]["input_sha256"], row["record"]["evidence_sha256"])
                for row in rows["tool_rows"]
            }
            records = await runtime.get_record_bundle(mission_id)
            observations = [record for record in records if isinstance(record, Observation)]
            findings = [record for record in records if isinstance(record, Finding)]
            assert len(observations) == 2
            assert [item.status for item in observations] == ["stale", "stale"]
            assert len(findings) == 1 and findings[0].visual_state == "unresolved"
            assert findings[0].review is not None
            assert (findings[0].review.state, findings[0].review.actor, findings[0].review.note) == (
                "accepted", "operator-2", "Reviewed historical result."
            )
            assert [item.frame_id for item in observations] == [1, 2]
            assert all((item.source_id, item.source_epoch) == (binding.source_id, binding.source_epoch)
                       for item in observations)
            assert {result.input_sha256 for item in observations for result in item.tool_results} == {
                value[0] for value in original.values()
            }
            assert {tuple(sorted(result.evidence_sha256.items())) for item in observations for result in item.tool_results} == {
                tuple(sorted(value[1].items())) for value in original.values()
            }
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
