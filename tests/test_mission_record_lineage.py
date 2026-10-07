"""Runtime regressions for stable finding and review lineage."""

import asyncio
import base64
import hashlib
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
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (32, 24), (12, 90, 175)).save(output, format="JPEG")
    return output.getvalue()


class _Planner:
    def __init__(self, choose):
        self.choose = choose
        self.calls = 0

    def available(self, model_key):
        return model_key == "gemma"

    def plan(self, context):
        self.calls += 1
        return self.choose(context)

    def close(self):
        pass


class _Tools:
    def __init__(self, on_detect=None):
        self.on_detect = on_detect

    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, request, _context):
        assert request.tool == "detect_objects"
        if self.on_detect is not None:
            self.on_detect()
        return ToolResult(
            "ok",
            items=(GeometryItem("container-1", "container", 0.9, (0.1, 0.2, 0.7, 0.8)),),
        )


class _Watch:
    def __init__(self):
        self.active = None

    def current_configuration_revision(self):
        return 0 if self.active is None else self.active.configuration_revision

    def apply(self, request):
        prior = self.active
        if prior is None:
            lease_id = "watch-lineage"
            revision = request.expected_configuration_revision + 1
        else:
            assert prior.mission_id == request.mission_id
            assert prior.source_binding == request.source_binding
            assert prior.configuration_revision == request.expected_configuration_revision
            lease_id = prior.lease_id
            revision = prior.configuration_revision
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


class _Frames:
    def __init__(self, binding):
        self.binding = binding
        self.frame_id = 1

    def advance(self):
        self.frame_id = 2

    def __call__(self, requested):
        if requested != self.binding:
            return None
        return SourceFrame(
            _jpeg(),
            self.binding.source_id,
            self.binding.source_epoch,
            self.frame_id,
            time.monotonic(),
        )


def _command(name, request_id, mission_id=None, revision=None, args=None):
    command = {
        "type": "mission_command",
        "schema_version": 1,
        "request_id": request_id,
        "command": name,
        "mission_id": mission_id,
        "args": args or {},
    }
    if revision is not None:
        command["expected_revision"] = revision
    return command


async def _wait_for(predicate, timeout=3.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_watch_cycle_history_uses_retained_finding_id_and_exact_frame_evidence(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("scout-lineage", "41")
    frames = _Frames(binding)

    def choose(context):
        if not context.tool_results:
            return Decision(1, "detect_objects", {"targets": ["container"]})
        record = context.tool_results[-1]
        return Decision(
            1,
            "finish",
            findings=(FindingProposal(
                "Container is present.",
                "localized_object",
                (),
                item_refs=((record.tool_result_id, "container-1"),),
            ),),
            watch=WatchProposal(("container",), "detect"),
        )

    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        watch = _Watch()
        planner = _Planner(choose)
        runtime = MissionRuntime(
            store,
            planner,
            _Tools(frames.advance),
            frames,
            watch,
            lambda _event: None,
            qualified_models={"gemma": ["watch"]},
        )
        try:
            created = await runtime.handle(
                _command(
                    "create",
                    "watch-create",
                    args={
                        "profile": {"id": "visual_inspection", "version": 1},
                        "expertise": "site inspector",
                        "mode": "watch",
                        "reasoning_model": "gemma",
                        "source_binding": {
                            "source_id": binding.source_id,
                            "source_epoch": binding.source_epoch,
                        },
                    },
                ),
                Principal("installation"),
                SCOPES,
            )
            assert created["ok"]
            initial = created["result"]["snapshot"]
            resumed = await runtime.handle(
                _command(
                    "resume",
                    "watch-resume",
                    initial["mission_id"],
                    initial["revision"],
                    {"source_binding": {
                        "source_id": binding.source_id,
                        "source_epoch": binding.source_epoch,
                    }},
                ),
                Principal("installation"),
                SCOPES,
            )
            assert resumed["ok"]
            mission_id = initial["mission_id"]

            def completed_two_cycles():
                current = store.get_mission(mission_id)
                if current["state"] == "failed":
                    raise AssertionError(f"Watch runtime failed before two cycles: {current}")
                return current if len(current.get("cycle_history", [])) >= 2 else None

            try:
                two_cycles = await _wait_for(completed_two_cycles)
            except AssertionError as exc:
                raise AssertionError(
                    f"Watch failed to complete cycles: {store.get_mission(mission_id)}; "
                    f"frame_id={frames.frame_id}; planner_calls={planner.calls}"
                ) from exc
            before_pause = store.get_mission(mission_id)
            await runtime.handle(
                _command("pause", "watch-pause", mission_id, before_pause["revision"]),
                Principal("installation"),
                SCOPES,
            )

            snapshot = store.get_mission(mission_id)
            finding = snapshot["findings"][0]
            canonical_id = finding["finding_id"]
            cycles = snapshot["cycle_history"][:2]
            evidence_by_id = {item["evidence_id"]: item for item in snapshot["evidence"]}
            cycle_evidence_ids = [item["evidence_refs"][0]["evidence_id"] for item in cycles]
            observations_by_evidence = {
                item["evidence_id"]: item for item in finding["observations"]
            }
            assert all(item["finding_ids"] == [canonical_id] for item in cycles)
            assert len(set(cycle_evidence_ids)) == 2
            cycle_frame_ids = [evidence_by_id[item]["frame_id"] for item in cycle_evidence_ids]
            assert cycle_frame_ids == [1, 2]
            for evidence_id in cycle_evidence_ids:
                observation = observations_by_evidence[evidence_id]
                assert observation["frame_id"] == evidence_by_id[evidence_id]["frame_id"]
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_review_snapshot_keeps_committed_revision_after_review_event_is_pruned(tmp_path):
    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            _Planner(lambda _context: Decision(
                1,
                "finish",
                findings=(FindingProposal(
                    "A visual condition needs review.", "visual_hypothesis", ()
                ),),
            )),
            _Tools(),
            lambda _binding: None,
            _Watch(),
            lambda _event: None,
            qualified_models={"gemma": ["inspect"]},
        )
        try:
            created = await runtime.handle(
                _command(
                    "create",
                    "inspect-create",
                    args={
                        "profile": {"id": "visual_inspection", "version": 1},
                        "expertise": "site inspector",
                        "mode": "inspect",
                        "reasoning_model": "gemma",
                    },
                ),
                Principal("installation"),
                SCOPES,
            )
            attached = await runtime.handle(
                _command(
                    "attach_evidence",
                    "inspect-attach",
                    created["result"]["snapshot"]["mission_id"],
                    created["result"]["snapshot"]["revision"],
                    {
                        "jpeg_b64": base64.b64encode(_jpeg()).decode(),
                        "sha256": hashlib.sha256(_jpeg()).hexdigest(),
                    },
                ),
                Principal("installation"),
                SCOPES,
            )
            assert attached["ok"]
            snapshot = attached["result"]["snapshot"]
            resumed = await runtime.handle(
                _command("resume", "inspect-resume", snapshot["mission_id"], snapshot["revision"]),
                Principal("installation"),
                SCOPES,
            )
            assert resumed["ok"]
            completed = await _wait_for(
                lambda: (
                    current
                    if (current := store.get_mission(snapshot["mission_id"]))["state"] == "completed"
                    else None
                )
            )
            finding = completed["findings"][0]
            visual_status, visual_reason = finding["status"], finding["reason"]
            review_time = time.time_ns() // 1_000_000
            reviewed = await runtime.handle(
                _command(
                    "review_finding",
                    "review-lineage",
                    snapshot["mission_id"],
                    completed["revision"],
                    {"finding_id": finding["finding_id"], "decision": "accepted", "note": "Checked."},
                ),
                Principal("operator-7"),
                SCOPES,
            )
            assert reviewed["ok"]
            committed = reviewed["result"]["snapshot"]
            committed_revision = committed["revision"]
            review = committed["findings"][0]["review"]

            def prune_review_event(tx):
                for sequence in range(205):
                    tx.append_event(
                        snapshot["mission_id"],
                        committed_revision,
                        "unrelated_test_event",
                        {"index": sequence},
                        review_time + sequence + 1,
                    )

            store.transact(prune_review_event)
            current = await runtime.handle(
                _command("get", "review-after-prune", snapshot["mission_id"]),
                Principal("installation"),
                SCOPES,
            )
            assert current["ok"]
            persisted = current["result"]["snapshot"]["findings"][0]
            retained_events = store.events_since(snapshot["mission_id"], 0)["events"]
            assert not any(event["kind"] == "finding_reviewed" for event in retained_events)
            assert review["actor"] == "operator-7"
            assert review["decision"] == "accepted"
            assert review["note"] == "Checked."
            assert isinstance(review["time_ms"], int)
            assert review["mission_revision"] == committed_revision
            assert persisted["review"] == review
            assert persisted["status"] == visual_status == "unresolved"
            assert persisted["reason"] == visual_reason
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
