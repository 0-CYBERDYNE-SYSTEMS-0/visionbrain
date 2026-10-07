"""CPU-only projections of persisted mission rows into released records."""

import asyncio
import base64
import copy
import hashlib
import json
import time
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
from visionbrain.mission_record_projection import (
    MissionRecordProjectionError,
    project_mission_records,
)
from visionbrain.mission_records import (
    EvidenceRecord,
    Finding,
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
    Image.new("RGB", (32, 24), (12, 90, 175)).save(output, format="JPEG")
    return output.getvalue()


class _Planner:
    def __init__(self, choose):
        self.choose = choose

    def available(self, model_key):
        return model_key == "gemma"

    def plan(self, context):
        return self.choose(context)

    def close(self):
        pass


class _Tools:
    def __init__(self, outcome="ok", on_detect=None):
        self.outcome = outcome
        self.on_detect = on_detect

    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, request, _context):
        if request.tool == "detect_objects":
            if self.on_detect is not None:
                self.on_detect()
            if self.outcome == "empty":
                return ToolResult("empty")
            if self.outcome == "failed":
                return ToolResult("failed", error_code="synthetic_detector_failure")
            if self.outcome == "ok_empty":
                return ToolResult("ok")
            return ToolResult(
                "ok",
                items=(GeometryItem("object-1", "container", 0.9, (0.1, 0.2, 0.7, 0.8)),),
            )
        if request.tool == "read_text":
            return ToolResult("ok", text="PUMP-27")
        raise AssertionError(f"unexpected fake tool: {request.tool}")


class _Watch:
    def __init__(self):
        self.active = None

    def current_configuration_revision(self):
        return 0 if self.active is None else self.active.configuration_revision

    def apply(self, request):
        prior = self.active
        if prior is None:
            revision, lease_id = request.expected_configuration_revision + 1, "projection-watch"
        else:
            assert prior.mission_id == request.mission_id
            assert prior.source_binding == request.source_binding
            assert prior.configuration_revision == request.expected_configuration_revision
            revision, lease_id = prior.configuration_revision, prior.lease_id
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
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def _tool_rows(store, mission_id):
    with store._lock:
        rows = store._connection.execute(
            "SELECT mission_id, cycle_id, execution_generation, is_current, record_json "
            "FROM tool_records WHERE mission_id = ? ORDER BY created_at_ms, record_id",
            (mission_id,),
        ).fetchall()
    return [
        {
            "mission_id": row[0],
            "cycle_id": row[1],
            "execution_generation": row[2],
            "is_current": bool(row[3]),
            "record": json.loads(row[4]),
        }
        for row in rows
    ]


def _evidence_rows(store, mission_id):
    with store._lock:
        rows = store._connection.execute(
            "SELECT mission_id, metadata_json, available FROM evidence "
            "WHERE mission_id = ? ORDER BY created_at_ms, evidence_id",
            (mission_id,),
        ).fetchall()
    output = []
    for row in rows:
        metadata = json.loads(row[1])
        metadata["mission_id"] = row[0]
        if not bool(row[2]):
            metadata["available"] = False
            metadata.setdefault("availability_reason", "rolled_off")
        else:
            metadata["available"] = True
        output.append(metadata)
    return output


def _inspect_planner(*, use_ocr=False):
    def choose(context):
        if not context.tool_results:
            return Decision(1, "detect_objects", {"targets": ["container"]})
        if use_ocr and len(context.tool_results) == 1:
            return Decision(1, "read_text", {"item_id": context.tool_results[0].items[0].item_id})
        return Decision(1, "finish")
    return _Planner(choose)


def _generation_projection_rows(current_generation=4, cycle_generation=3):
    mission_id = "generation-projection"
    cycle_id = "cycle-generation-1"
    evidence_id = "evidence-generation-1"
    snapshot = {
        "mission_id": mission_id,
        "mode": "inspect",
        "execution_generation": current_generation,
        "cycle_history": [{
            "cycle_id": cycle_id,
            "execution_generation": cycle_generation,
            "outcome": "completed",
            "evidence_refs": [{"evidence_id": evidence_id}],
            "finding_ids": [],
        }],
        "findings": [],
    }
    evidence_rows = [{
        "mission_id": mission_id,
        "evidence_id": evidence_id,
        "sha256": "a" * 64,
        "width": 32,
        "height": 24,
        "kind": "original",
        "bytes": 1,
        "available": True,
    }]
    tool_rows = [{
        "mission_id": mission_id,
        "cycle_id": cycle_id,
        "execution_generation": cycle_generation,
        "is_current": True,
        "record": {
            "tool_result_id": "tool-generation-1",
            "tool": "detect_objects",
            "status": "ok",
            "input_evidence_id": evidence_id,
            "items": [],
            "evidence_ids": [],
            "text": "",
        },
    }]
    return snapshot, evidence_rows, tool_rows


@pytest.mark.parametrize("current_generation", [True, "4", -1])
def test_present_malformed_snapshot_generation_is_rejected(current_generation):
    snapshot, evidence_rows, tool_rows = _generation_projection_rows(current_generation)

    with pytest.raises(MissionRecordProjectionError, match="invalid_execution_generation"):
        project_mission_records(snapshot, evidence_rows, tool_rows)


def test_cycle_generation_ahead_of_snapshot_is_rejected_as_invalid_history():
    snapshot, evidence_rows, tool_rows = _generation_projection_rows(current_generation=2, cycle_generation=3)

    with pytest.raises(MissionRecordProjectionError, match="cycle_generation_ahead_of_snapshot"):
        project_mission_records(snapshot, evidence_rows, tool_rows)


def test_absent_snapshot_generation_preserves_legacy_projection_behavior():
    snapshot, evidence_rows, tool_rows = _generation_projection_rows(current_generation=4)
    snapshot.pop("execution_generation")

    records = project_mission_records(snapshot, evidence_rows, tool_rows)

    observation = next(record for record in records if isinstance(record, Observation))
    assert (observation.status, observation.outcome) == ("observed", "empty")


@pytest.mark.parametrize(
    ("tool_mode", "tool_status", "observation_status", "observation_outcome", "error_code", "append_empty"),
    [
        ("empty", "empty", "observed", "empty", None, False),
        ("ok_empty", "ok", "observed", "empty", None, True),
        ("ok", "ok", "observed", "nonempty", None, True),
        ("failed", "failed", "failed", "failed", "synthetic_detector_failure", True),
    ],
)
def test_inspect_projects_explicit_empty_and_failed_tool_status(
    tmp_path, tool_mode, tool_status, observation_status, observation_outcome, error_code, append_empty
):
    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            _inspect_planner(),
            _Tools(tool_mode),
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
            resumed = await runtime.handle(
                _command("resume", "inspect-resume", snapshot["mission_id"], snapshot["revision"]),
                Principal("installation"),
                SCOPES,
            )
            assert resumed["ok"]
            await _wait_for(lambda: (
                current if (current := store.get_mission(snapshot["mission_id"]))["state"] == "completed" else None
            ))
            final = store.get_mission(snapshot["mission_id"])
            evidence_rows = _evidence_rows(store, snapshot["mission_id"])
            tool_rows = _tool_rows(store, snapshot["mission_id"])
            if append_empty:
                empty_row = copy.deepcopy(tool_rows[0])
                empty_record = empty_row["record"]
                empty_record.update({
                    "tool_result_id": empty_record["tool_result_id"] + "-empty",
                    "status": "empty",
                    "items": [],
                    "evidence_ids": [],
                    "unavailable_evidence_ids": [],
                    "text": "",
                    "error_code": None,
                })
                tool_rows.append(empty_row)
            records = project_mission_records(final, evidence_rows, tool_rows)
            observation = next(record for record in records if isinstance(record, Observation))
            assert observation.status == observation_status
            assert observation.outcome == observation_outcome
            assert observation.error_code == error_code
            assert [result.status for result in observation.tool_results] == (
                [tool_status, "empty"] if append_empty else [tool_status]
            )
            validate_record_bundle(records)
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_inspect_ocr_projection_preserves_explicit_result_and_unknown_optional_times(tmp_path):
    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            _inspect_planner(use_ocr=True),
            _Tools(),
            lambda _binding: None,
            _Watch(),
            lambda _event: None,
            qualified_models={"gemma": ["inspect"]},
        )
        try:
            created = await runtime.handle(
                _command("create", "ocr-create", args={
                    "profile": {"id": "visual_inspection", "version": 1},
                    "expertise": "site inspector",
                    "mode": "inspect",
                    "reasoning_model": "gemma",
                }),
                Principal("installation"),
                SCOPES,
            )
            initial = created["result"]["snapshot"]
            jpeg = _jpeg()
            attached = await runtime.handle(
                _command("attach_evidence", "ocr-evidence", initial["mission_id"], initial["revision"], {
                    "jpeg_b64": base64.b64encode(jpeg).decode(),
                    "sha256": hashlib.sha256(jpeg).hexdigest(),
                }),
                Principal("installation"),
                SCOPES,
            )
            snapshot = attached["result"]["snapshot"]
            await runtime.handle(
                _command("resume", "ocr-resume", snapshot["mission_id"], snapshot["revision"]),
                Principal("installation"),
                SCOPES,
            )
            await _wait_for(lambda: (
                current if (current := store.get_mission(snapshot["mission_id"]))["state"] == "completed" else None
            ))
            raw_snapshot = store.get_mission(snapshot["mission_id"])
            evidence_rows = _evidence_rows(store, snapshot["mission_id"])
            tool_rows = _tool_rows(store, snapshot["mission_id"])
            original = copy.deepcopy((raw_snapshot, evidence_rows, tool_rows))
            records = project_mission_records(raw_snapshot, evidence_rows, tool_rows)
            assert (raw_snapshot, evidence_rows, tool_rows) == original
            missing_evidence_owner = copy.deepcopy(evidence_rows)
            missing_evidence_owner[0].pop("mission_id")
            with pytest.raises(MissionRecordProjectionError, match="missing_required_fact.*mission_id"):
                project_mission_records(raw_snapshot, missing_evidence_owner, tool_rows)
            cross_mission_tool = copy.deepcopy(tool_rows)
            cross_mission_tool[0]["mission_id"] = "another-mission"
            with pytest.raises(MissionRecordProjectionError, match="tool_mission_mismatch"):
                project_mission_records(raw_snapshot, evidence_rows, cross_mission_tool)
            text_finding_snapshot = copy.deepcopy(raw_snapshot)
            text_finding_id = "projected-text-finding"
            cited_result_id = next(
                row["record"]["tool_result_id"]
                for row in tool_rows
                if row["record"]["tool"] == "read_text"
            )
            input_evidence_id = text_finding_snapshot["cycle_history"][0]["evidence_refs"][0]["evidence_id"]
            text_finding_snapshot["findings"].append({
                "finding_id": text_finding_id,
                "claim": "PUMP-27",
                "claim_type": "text_read",
                "status": "supported",
                "reason": "exact_ocr_transcription_only",
                "evidence_id": input_evidence_id,
                "evidence_refs": [input_evidence_id],
                "text_refs": [cited_result_id],
                "item_refs": [],
                "items": [],
                "localization": None,
                "source_binding": None,
                "brief_version": 1,
                "brief_sha256": None,
                "model_provenance": {},
                "review": None,
            })
            text_finding_snapshot["cycle_history"][0]["finding_ids"].append(text_finding_id)
            text_records = project_mission_records(text_finding_snapshot, evidence_rows, tool_rows)
            projected_finding = next(
                record for record in text_records
                if isinstance(record, Finding) and record.finding_id == text_finding_id
            )
            assert projected_finding.record_schema_version == 2
            assert projected_finding.text_refs == (cited_result_id,)
            assert projected_finding.claim_type == "text_read"
            assert projected_finding.visual_state == "supported"
            assert projected_finding.reason == "exact_ocr_transcription_only"
            projected_observation = next(record for record in text_records if isinstance(record, Observation))
            assert next(result for result in projected_observation.tool_results if result.tool_result_id == cited_result_id).text == "PUMP-27"
            missing_text_refs = copy.deepcopy(text_finding_snapshot)
            missing_text_refs["findings"][-1]["text_refs"] = []
            with pytest.raises(MissionRecordProjectionError, match="text_finding_citations_missing"):
                project_mission_records(missing_text_refs, evidence_rows, tool_rows)
            observation = next(record for record in records if isinstance(record, Observation))
            assert observation.status == "observed"
            assert observation.outcome == "nonempty"
            assert [item.tool for item in observation.tool_results] == ["detect_objects", "read_text"]
            assert observation.tool_results[1].status == "ok"
            assert observation.tool_results[1].text == "PUMP-27"
            assert observation.capture_time_ms is None
            assert observation.capture_time_quality == "unknown"
            assert observation.received_at_ms is None
            evidence_record = next(record for record in records if isinstance(record, EvidenceRecord))
            assert evidence_record.evidence.input_transform is None
            # A successful crop with a persisted child-evidence reference is output
            # even when it returns no geometry or text.
            crop_tool = copy.deepcopy(tool_rows[0])
            crop_record = crop_tool["record"]
            crop_evidence_id = "projection-crop-evidence"
            crop_record.update({
                "tool_result_id": crop_record["tool_result_id"] + "-crop",
                "tool": "inspect_crop",
                "items": [],
                "evidence_ids": [crop_evidence_id],
                "text": "",
                "evidence_sha256": {crop_evidence_id: evidence_rows[0]["sha256"]},
            })
            crop_evidence = copy.deepcopy(evidence_rows[0])
            crop_evidence.update({
                "evidence_id": crop_evidence_id,
                "kind": "crop",
                "parent_evidence_id": evidence_rows[0]["evidence_id"],
                "crop_box": [0.0, 0.0, 1.0, 1.0],
                "input_transform": None,
            })
            crop_records = project_mission_records(
                raw_snapshot,
                [*evidence_rows, crop_evidence],
                [crop_tool],
            )
            crop_observation = next(record for record in crop_records if isinstance(record, Observation))
            assert crop_observation.outcome == "nonempty"
            assert crop_observation.tool_results[0].evidence_ids == (crop_evidence_id,)
            round_trip = tuple(deserialize_record(serialize_record(record)) for record in records)
            assert round_trip == records
            validate_record_bundle(round_trip)
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_watch_projection_links_two_frames_to_retained_finding_and_review_revision(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    binding = SourceBinding("scout-projection", "41")
    frames = _Frames(binding)

    def choose(context):
        if not context.tool_results:
            return Decision(1, "detect_objects", {"targets": ["container"]})
        tool = context.tool_results[-1]
        return Decision(
            1,
            "finish",
            findings=(FindingProposal(
                "Container is present.", "localized_object", (),
                item_refs=((tool.tool_result_id, "object-1"),),
            ),),
            watch=WatchProposal(("container",), "detect"),
        )

    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            _Planner(choose),
            _Tools(on_detect=frames.advance),
            frames,
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
            initial = created["result"]["snapshot"]
            resumed = await runtime.handle(
                _command("resume", "watch-resume", initial["mission_id"], initial["revision"], {
                    "source_binding": {"source_id": binding.source_id, "source_epoch": binding.source_epoch},
                }),
                Principal("installation"),
                SCOPES,
            )
            assert resumed["ok"]
            mission_id = initial["mission_id"]
            await _wait_for(lambda: (
                current if len(current := store.get_mission(mission_id)["cycle_history"]) >= 2 else None
            ))
            before_pause = store.get_mission(mission_id)
            paused = await runtime.handle(
                _command("pause", "watch-pause", mission_id, before_pause["revision"]),
                Principal("installation"),
                SCOPES,
            )
            assert paused["ok"]
            snapshot = store.get_mission(mission_id)
            finding = snapshot["findings"][0]
            reviewed = await runtime.handle(
                _command("review_finding", "watch-review", mission_id, snapshot["revision"], {
                    "finding_id": finding["finding_id"], "decision": "accepted", "note": "Verified visually.",
                }),
                Principal("operator-2"),
                SCOPES,
            )
            assert reviewed["ok"]
            snapshot = store.get_mission(mission_id)
            evidence_rows = _evidence_rows(store, mission_id)
            tool_rows = _tool_rows(store, mission_id)
            records = project_mission_records(snapshot, evidence_rows, tool_rows)
            observations = [record for record in records if isinstance(record, Observation)]
            evidence = {record.evidence.evidence_id: record for record in records if isinstance(record, EvidenceRecord)}
            projected_finding = next(record for record in records if isinstance(record, Finding))
            assert len(observations) == 2
            assert [record.frame_id for record in observations] == [1, 2]
            assert all((record.source_id, record.source_epoch) == (binding.source_id, binding.source_epoch) for record in observations)
            cycle_evidence_ids = [cycle["evidence_refs"][0]["evidence_id"] for cycle in snapshot["cycle_history"]]
            assert [record.evidence_ids[0] for record in observations] == cycle_evidence_ids
            assert [evidence[item].frame_id for item in cycle_evidence_ids] == [1, 2]
            assert projected_finding.finding_id == finding["finding_id"]
            assert projected_finding.observation_ids == tuple(record.observation_id for record in observations)
            assert projected_finding.review is not None
            assert projected_finding.review.mission_revision == snapshot["revision"]
            assert projected_finding.visual_state == finding["status"] == "unresolved"
            round_trip = tuple(deserialize_record(serialize_record(record)) for record in records)
            assert round_trip == records
            validate_record_bundle(round_trip)
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_projection_rejects_missing_watch_source_epoch_and_review_revision(tmp_path):
    async def scenario():
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            _inspect_planner(),
            _Tools(),
            lambda _binding: None,
            _Watch(),
            lambda _event: None,
            qualified_models={"gemma": ["inspect"]},
        )
        try:
            created = await runtime.handle(
                _command("create", "error-create", args={
                    "profile": {"id": "visual_inspection", "version": 1},
                    "expertise": "site inspector", "mode": "inspect", "reasoning_model": "gemma",
                }),
                Principal("installation"), SCOPES,
            )
            initial = created["result"]["snapshot"]
            jpeg = _jpeg()
            attached = await runtime.handle(
                _command("attach_evidence", "error-evidence", initial["mission_id"], initial["revision"], {
                    "jpeg_b64": base64.b64encode(jpeg).decode(), "sha256": hashlib.sha256(jpeg).hexdigest(),
                }),
                Principal("installation"), SCOPES,
            )
            snapshot = attached["result"]["snapshot"]
            await runtime.handle(
                _command("resume", "error-resume", snapshot["mission_id"], snapshot["revision"]),
                Principal("installation"), SCOPES,
            )
            await _wait_for(lambda: (
                current if (current := store.get_mission(snapshot["mission_id"]))["state"] == "completed" else None
            ))
            final = store.get_mission(snapshot["mission_id"])
            evidence_rows = _evidence_rows(store, snapshot["mission_id"])
            tool_rows = _tool_rows(store, snapshot["mission_id"])
            evidence_rows[0]["capture_time_ms"] = 1_800_000_000_000
            with pytest.raises(MissionRecordProjectionError, match="capture_time_provenance_missing"):
                project_mission_records(final, evidence_rows, tool_rows)
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
