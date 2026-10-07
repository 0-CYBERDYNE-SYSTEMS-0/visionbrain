"""Pure projection of completed mission snapshots into released records."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from .mission_contracts import (
    EvidenceRef,
    GeometryItem,
    InputTransform,
    SourceBinding,
    ToolCallRecord,
)
from .mission_records import (
    EvidenceRecord,
    FINDING_TEXT_CITATION_SCHEMA_VERSION,
    Finding,
    Localization,
    LocalizationStatement,
    Observation,
    Record,
    RecordValidationError,
    Review,
    validate_record_bundle,
)


class MissionRecordProjectionError(ValueError):
    """A persisted mission fact cannot be represented without inference."""

    def __init__(self, reason: str, path: str = "") -> None:
        self.reason = reason
        self.path = path
        super().__init__(f"{reason}{': ' + path if path else ''}")


def _fail(reason: str, path: str) -> None:
    raise MissionRecordProjectionError(reason, path)


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("expected_object", path)
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        _fail("missing_required_fact", f"{path}.{key}")
    return mapping[key]


def _text(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _required(mapping, key, path)
    if not isinstance(value, str) or not value:
        _fail("invalid_text", f"{path}.{key}")
    return value


def _rows(value: Any, path: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _fail("expected_rows", path)
    return [_object(item, f"{path}[{index}]") for index, item in enumerate(value)]


def _model_provenance(value: Any, path: str) -> tuple[tuple[str, Any], ...]:
    data = {} if value is None else _object(value, path)
    output = []
    for key, item in sorted(data.items()):
        if not isinstance(key, str) or not isinstance(item, (str, int, float, bool, type(None))):
            _fail("unrepresentable_model_provenance", path)
        output.append((key, item))
    return tuple(output)


def _binding(value: Any, path: str) -> SourceBinding | None:
    if value is None:
        return None
    data = _object(value, path)
    if not isinstance(data.get("source_id"), str) or not isinstance(data.get("source_epoch"), str):
        _fail("incomplete_source_binding", path)
    return SourceBinding(data["source_id"], data["source_epoch"])


def _geometry(value: Any, path: str) -> GeometryItem:
    data = _object(value, path)
    box = _required(data, "box", path)
    polygon = data.get("polygon")
    if not isinstance(box, (list, tuple)):
        _fail("invalid_geometry", f"{path}.box")
    if polygon is not None:
        if not isinstance(polygon, (list, tuple)):
            _fail("invalid_polygon", f"{path}.polygon")
        polygon = tuple(tuple(point) if isinstance(point, (list, tuple)) else point for point in polygon)
    return GeometryItem(
        item_id=_text(data, "item_id", path),
        label=_text(data, "label", path),
        score=_required(data, "score", path),
        box=tuple(box),
        polygon=polygon,
        source=data.get("source", "visionbrain"),
    )


def _tool_call(value: Any, path: str) -> ToolCallRecord:
    data = _object(value, path)
    raw_items = data.get("items", ())
    raw_evidence = data.get("evidence_ids", ())
    raw_unavailable = data.get("unavailable_evidence_ids", ())
    for name, raw in (("items", raw_items), ("evidence_ids", raw_evidence), ("unavailable_evidence_ids", raw_unavailable)):
        if not isinstance(raw, (list, tuple)):
            _fail("expected_sequence", f"{path}.{name}")
    binding = data.get("source_binding")
    if binding is not None:
        binding = dict(_object(binding, f"{path}.source_binding"))
    provenance = data.get("model_provenance", {})
    evidence_hashes = data.get("evidence_sha256", {})
    if not isinstance(provenance, Mapping) or not isinstance(evidence_hashes, Mapping):
        _fail("expected_mapping", path)
    return ToolCallRecord(
        tool_result_id=_text(data, "tool_result_id", path),
        tool=_text(data, "tool", path),
        status=_text(data, "status", path),
        input_evidence_id=_text(data, "input_evidence_id", path),
        items=tuple(_geometry(item, f"{path}.items[{index}]") for index, item in enumerate(raw_items)),
        evidence_ids=tuple(raw_evidence),
        text=data.get("text", ""),
        error_code=data.get("error_code"),
        brief_version=data.get("brief_version", 1),
        brief_sha256=data.get("brief_sha256"),
        model_provenance=dict(provenance),
        unavailable_evidence_ids=tuple(raw_unavailable),
        source_binding=binding,
        frame_id=data.get("frame_id"),
        input_sha256=data.get("input_sha256"),
        evidence_sha256=dict(evidence_hashes),
        created_at_ms=data.get("created_at_ms"),
    )


def _input_evidence_id(cycle: Mapping[str, Any], path: str) -> str:
    refs = _rows(_required(cycle, "evidence_refs", path), f"{path}.evidence_refs")
    if len(refs) != 1:
        _fail("cycle_input_evidence_ambiguous", f"{path}.evidence_refs")
    return _text(refs[0], "evidence_id", f"{path}.evidence_refs[0]")


def _availability(row: Mapping[str, Any], path: str) -> tuple[str, str | None]:
    available = _required(row, "available", path)
    if type(available) is not bool:
        _fail("invalid_availability", f"{path}.available")
    if available:
        return "available", None
    reason = row.get("availability_reason")
    if not isinstance(reason, str) or reason not in {"missing", "corrupt", "rolled_off"}:
        _fail("unclassified_evidence_unavailability", f"{path}.availability_reason")
    return reason, reason


def _capture(row: Mapping[str, Any], path: str) -> tuple[int | None, str | None, str]:
    value = row.get("capture_time_ms")
    provenance = row.get("capture_time_provenance")
    quality = row.get("capture_time_quality", "unknown")
    if value is None:
        if provenance is not None or quality != "unknown":
            _fail("capture_time_provenance_mismatch", path)
        return None, None, "unknown"
    if provenance is None:
        _fail("capture_time_provenance_missing", path)
    return value, provenance, quality


def _evidence_row(row: Mapping[str, Any], mission_id: str, source_observation_id: str | None, path: str) -> EvidenceRecord:
    evidence_id = _text(row, "evidence_id", path)
    availability, availability_reason = _availability(row, path)
    capture_ms, capture_provenance, capture_quality = _capture(row, path)
    transform = row.get("input_transform")
    if transform is not None:
        transform_data = dict(_object(transform, f"{path}.input_transform"))
        if not transform_data:
            transform = None
        else:
            try:
                transform = InputTransform(**transform_data)
            except (TypeError, ValueError) as exc:
                raise MissionRecordProjectionError("invalid_input_transform", f"{path}.input_transform") from exc
    crop_box = row.get("crop_box")
    if crop_box is not None:
        if not isinstance(crop_box, (list, tuple)):
            _fail("invalid_crop_box", f"{path}.crop_box")
        crop_box = tuple(crop_box)
    return EvidenceRecord(
        mission_id=mission_id,
        evidence=EvidenceRef(
            evidence_id=evidence_id,
            sha256=_text(row, "sha256", path),
            width=_required(row, "width", path),
            height=_required(row, "height", path),
            kind=_text(row, "kind", path),
            parent_evidence_id=row.get("parent_evidence_id"),
            crop_box=crop_box,
            input_transform=transform,
        ),
        byte_length=_required(row, "bytes", path),
        availability=availability,
        availability_reason=availability_reason,
        source_observation_id=source_observation_id,
        source_id=row.get("source_id"),
        source_epoch=row.get("source_epoch"),
        frame_id=row.get("frame_id"),
        capture_time_ms=capture_ms,
        capture_time_provenance=capture_provenance,
        capture_time_quality=capture_quality,
        created_at_ms=row.get("created_at_ms"),
        closeup_request_id=row.get("closeup_request_id"),
        brief_version=row.get("brief_version"),
        brief_sha256=row.get("brief_sha256"),
        model_provenance=_model_provenance(row.get("model_provenance", {}), f"{path}.model_provenance"),
    )


def _outcome(results: tuple[ToolCallRecord, ...], path: str) -> tuple[str, str, str | None]:
    failures = [result for result in results if result.status in {"failed", "timeout", "unsupported"}]
    if failures:
        first = failures[0]
        return "failed", "failed", first.error_code or first.status
    if any(result.status not in {"empty", "ok"} for result in results):
        _fail("unsupported_tool_status", path)
    successful_output = [
        result for result in results
        if result.status == "ok" and (result.items or result.evidence_ids or result.text.strip())
    ]
    if successful_output:
        return "observed", "nonempty", None
    if results and all(result.status in {"empty", "ok"} for result in results):
        return "observed", "empty", None
    if not results:
        return "unavailable", "not_run", None


def _localization(value: Any, path: str) -> Localization | None:
    if value is None:
        return None
    data = _object(value, path)
    statements = _rows(_required(data, "statements", path), f"{path}.statements")
    output = []
    for index, statement in enumerate(statements):
        positions = _rows(statement.get("positions", ()), f"{path}.statements[{index}].positions")
        output.append(LocalizationStatement(
            claim=_text(statement, "claim", f"{path}.statements[{index}]"),
            label=_text(statement, "label", f"{path}.statements[{index}]"),
            count=_required(statement, "count", f"{path}.statements[{index}]"),
            positions=tuple(
                (position.get("x"), position.get("y"))
                for position in positions
            ),
        ))
    return Localization(
        status=_text(data, "status", path),
        basis=_text(data, "basis", path),
        evidence_id=_text(data, "evidence_id", path),
        statements=tuple(output),
    )


def _review(value: Any, finding_id: str, path: str) -> Review | None:
    if value is None:
        return None
    data = _object(value, path)
    decision = _text(data, "decision", path)
    if decision not in {"accepted", "corrected", "rejected"}:
        _fail("unsupported_review_decision", f"{path}.decision")
    revision = data.get("mission_revision")
    if revision is None:
        _fail("review_revision_missing", f"{path}.mission_revision")
    return Review(
        finding_id=finding_id,
        state=decision,
        actor=_text(data, "actor", path),
        reviewed_at_ms=_required(data, "time_ms", path),
        mission_revision=revision,
        note=data.get("note"),
    )


def project_mission_records(
    snapshot: Mapping[str, Any],
    evidence_rows: Sequence[Mapping[str, Any]],
    tool_rows: Sequence[Mapping[str, Any]],
) -> tuple[Record, ...]:
    """Project one mission's complete persisted rows without mutating inputs.

    Each evidence row carries its persisted ``mission_id``. Each tool row has
    ``mission_id``, ``cycle_id``, ``execution_generation``, ``is_current``, and a
    nested ``record`` mapping containing persisted ToolCallRecord JSON. Rows must
    cover all retained evidence and tool records for this mission.
    """
    data = _object(snapshot, "snapshot")
    mission_id = _text(data, "mission_id", "snapshot")
    mode = _text(data, "mode", "snapshot")
    if mode not in {"inspect", "watch"}:
        _fail("unsupported_mission_mode", "snapshot.mode")
    current_generation = data.get("execution_generation")
    if "execution_generation" in data and (
        isinstance(current_generation, bool)
        or not isinstance(current_generation, int)
        or current_generation < 0
    ):
        _fail("invalid_execution_generation", "snapshot.execution_generation")
    if "execution_generation" not in data:
        current_generation = None
    if data.get("cycle_id") is not None:
        _fail("active_cycle_not_projectable", "snapshot.cycle_id")

    raw_cycles = _rows(_required(data, "cycle_history", "snapshot"), "snapshot.cycle_history")
    if not raw_cycles:
        _fail("no_completed_cycles", "snapshot.cycle_history")
    cycles: dict[str, Mapping[str, Any]] = {}
    input_ids: dict[str, str] = {}
    for index, cycle in enumerate(raw_cycles):
        path = f"snapshot.cycle_history[{index}]"
        cycle_id = _text(cycle, "cycle_id", path)
        if cycle_id in cycles:
            _fail("duplicate_cycle_id", f"{path}.cycle_id")
        if cycle.get("outcome") != "completed":
            _fail("cycle_not_completed", f"{path}.outcome")
        generation = _required(cycle, "execution_generation", path)
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            _fail("invalid_execution_generation", f"{path}.execution_generation")
        if current_generation is not None and generation > current_generation:
            _fail("cycle_generation_ahead_of_snapshot", f"{path}.execution_generation")
        input_ids[cycle_id] = _input_evidence_id(cycle, path)
        cycles[cycle_id] = cycle

    raw_evidence = _rows(evidence_rows, "evidence_rows")
    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(raw_evidence):
        path = f"evidence_rows[{index}]"
        evidence_id = _text(row, "evidence_id", path)
        if evidence_id in evidence_by_id:
            _fail("duplicate_evidence_id", f"{path}.evidence_id")
        row_mission_id = _text(row, "mission_id", path)
        if row_mission_id != mission_id:
            _fail("evidence_mission_mismatch", path)
        evidence_by_id[evidence_id] = row

    raw_tool_rows = _rows(tool_rows, "tool_rows")
    tools_by_cycle: dict[str, list[ToolCallRecord]] = {cycle_id: [] for cycle_id in cycles}
    for index, row in enumerate(raw_tool_rows):
        path = f"tool_rows[{index}]"
        if _text(row, "mission_id", path) != mission_id:
            _fail("tool_mission_mismatch", path)
        cycle_id = _text(row, "cycle_id", path)
        cycle = cycles.get(cycle_id)
        if cycle is None:
            _fail("tool_row_has_no_retained_cycle", f"{path}.cycle_id")
        current = _required(row, "is_current", path)
        if not (current is True or (type(current) is int and current == 1)):
            _fail("noncurrent_tool_result", f"{path}.is_current")
        generation = _required(row, "execution_generation", path)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or generation != cycle.get("execution_generation")
        ):
            _fail("tool_cycle_generation_mismatch", path)
        record = _tool_call(_required(row, "record", path), f"{path}.record")
        if record.input_evidence_id != input_ids[cycle_id]:
            _fail("tool_input_evidence_mismatch", path)
        tools_by_cycle[cycle_id].append(record)

    observations: list[Observation] = []
    observation_by_cycle: dict[str, Observation] = {}
    observation_for_evidence: dict[str, set[str]] = {}
    for index, (cycle_id, cycle) in enumerate(cycles.items()):
        path = f"snapshot.cycle_history[{index}]"
        evidence_id = input_ids[cycle_id]
        input_row = evidence_by_id.get(evidence_id)
        if input_row is None:
            _fail("cycle_input_evidence_missing", evidence_id)
        results = tuple(tools_by_cycle[cycle_id])
        status, outcome, error_code = _outcome(results, f"{path}.tool_results")
        generation = cycle["execution_generation"]
        # Retained Watch results from an older execution cannot imply current support.
        if current_generation is not None and generation < current_generation:
            status = "stale"
        referenced_ids = [evidence_id]
        for result in results:
            if result.input_evidence_id != evidence_id:
                _fail("tool_input_evidence_mismatch", f"{path}.tool_results")
            referenced_ids.extend(result.evidence_ids)
            referenced_ids.extend(result.unavailable_evidence_ids)
        evidence_ids = tuple(dict.fromkeys(referenced_ids))
        for linked_id in evidence_ids:
            if linked_id not in evidence_by_id:
                _fail("tool_evidence_row_missing", linked_id)
        source_id = input_row.get("source_id")
        source_epoch = input_row.get("source_epoch")
        frame_id = input_row.get("frame_id")
        if mode == "watch" and (not source_id or not source_epoch or frame_id is None):
            _fail("watch_source_frame_provenance_missing", evidence_id)
        capture_ms, capture_provenance, capture_quality = _capture(input_row, f"evidence_rows[{evidence_id}]")
        observation_id = "obs-" + hashlib.sha256(f"{mission_id}\0{cycle_id}".encode()).hexdigest()[:32]
        observation = Observation(
            observation_id=observation_id,
            mission_id=mission_id,
            status=status,
            outcome=outcome,
            source_id=source_id,
            source_epoch=source_epoch,
            frame_id=frame_id,
            width=input_row.get("width"),
            height=input_row.get("height"),
            evidence_ids=evidence_ids,
            tool_results=results,
            error_code=error_code,
            capture_time_ms=capture_ms,
            capture_time_provenance=capture_provenance,
            capture_time_quality=capture_quality,
        )
        observations.append(observation)
        observation_by_cycle[cycle_id] = observation
        for linked_id in evidence_ids:
            observation_for_evidence.setdefault(linked_id, set()).add(observation_id)

    evidence_records = []
    for index, (evidence_id, row) in enumerate(evidence_by_id.items()):
        source_observations = observation_for_evidence.get(evidence_id, set())
        if len(source_observations) > 1:
            _fail("ambiguous_evidence_observation", f"evidence_rows[{index}].evidence_id")
        source_observation_id = next(iter(source_observations), None)
        evidence_records.append(_evidence_row(
            row,
            mission_id,
            source_observation_id,
            f"evidence_rows[{index}]",
        ))

    raw_findings = _rows(_required(data, "findings", "snapshot"), "snapshot.findings")
    findings_by_id: dict[str, Mapping[str, Any]] = {}
    observation_ids_by_finding: dict[str, list[str]] = {}
    for index, finding in enumerate(raw_findings):
        path = f"snapshot.findings[{index}]"
        finding_id = _text(finding, "finding_id", path)
        if finding_id in findings_by_id:
            _fail("duplicate_finding_id", f"{path}.finding_id")
        findings_by_id[finding_id] = finding
        observation_ids_by_finding[finding_id] = []

    for index, (cycle_id, cycle) in enumerate(cycles.items()):
        path = f"snapshot.cycle_history[{index}]"
        observation = observation_by_cycle[cycle_id]
        for finding_id in _required(cycle, "finding_ids", path):
            finding = findings_by_id.get(finding_id)
            if finding is None:
                _fail("cycle_finding_missing", f"{path}.finding_ids:{finding_id}")
            input_id = input_ids[cycle_id]
            refs = set(finding.get("evidence_refs", ())) | {finding.get("evidence_id")}
            if input_id not in refs:
                _fail("finding_cycle_evidence_mismatch", f"{path}.finding_ids:{finding_id}")
            if mode == "watch":
                history = _rows(finding.get("observations", ()), f"finding[{finding_id}].observations")
                matching = [item for item in history if item.get("evidence_id") == input_id]
                if len(matching) != 1:
                    _fail("watch_finding_observation_mismatch", f"finding[{finding_id}].observations")
                source = _object(matching[0].get("source_binding"), f"finding[{finding_id}].observations.source_binding")
                if (
                    matching[0].get("frame_id") != observation.frame_id
                    or source.get("source_id") != observation.source_id
                    or source.get("source_epoch") != observation.source_epoch
                ):
                    _fail("watch_finding_source_frame_mismatch", f"finding[{finding_id}].observations")
            observation_ids_by_finding[finding_id].append(observation.observation_id)

    findings: list[Finding] = []
    for index, raw in enumerate(raw_findings):
        path = f"snapshot.findings[{index}]"
        finding_id = findings_by_id[raw["finding_id"]]["finding_id"]
        refs = raw.get("evidence_refs", ())
        if not isinstance(refs, (list, tuple)):
            _fail("expected_sequence", f"{path}.evidence_refs")
        claim_type = _text(raw, "claim_type", path)
        text_refs = raw.get("text_refs", ())
        if not isinstance(text_refs, (list, tuple)):
            _fail("expected_sequence", f"{path}.text_refs")
        if claim_type == "text_read":
            if not text_refs:
                _fail("text_finding_citations_missing", f"{path}.text_refs")
            record_schema_version = FINDING_TEXT_CITATION_SCHEMA_VERSION
        elif text_refs:
            _fail("text_refs_claim_type_mismatch", f"{path}.text_refs")
        else:
            record_schema_version = 1
        item_refs = raw.get("item_refs", ())
        if not isinstance(item_refs, (list, tuple)):
            _fail("expected_sequence", f"{path}.item_refs")
        visual_state = _text(raw, "status", path)
        if visual_state not in {"candidate", "supported", "unresolved"}:
            _fail("unsupported_visual_state", f"{path}.status")
        findings.append(Finding(
            finding_id=finding_id,
            mission_id=mission_id,
            claim=_text(raw, "claim", path),
            claim_type=claim_type,
            visual_state=visual_state,
            reason=_text(raw, "reason", path),
            evidence_id=_text(raw, "evidence_id", path),
            evidence_refs=tuple(refs),
            text_refs=tuple(text_refs),
            observation_ids=tuple(dict.fromkeys(observation_ids_by_finding[finding_id])),
            item_refs=tuple(
                (_text(_object(item, f"{path}.item_refs[{item_index}]"), "tool_result_id", f"{path}.item_refs[{item_index}]"),
                 _text(_object(item, f"{path}.item_refs[{item_index}]"), "item_id", f"{path}.item_refs[{item_index}]"))
                for item_index, item in enumerate(item_refs)
            ),
            items=tuple(_geometry(item, f"{path}.items[{item_index}]") for item_index, item in enumerate(_rows(raw.get("items", ()), f"{path}.items"))),
            localization=_localization(raw.get("localization"), f"{path}.localization"),
            source_binding=_binding(raw.get("source_binding"), f"{path}.source_binding"),
            frame_id=raw.get("frame_id"),
            brief_version=raw.get("brief_version"),
            brief_sha256=raw.get("brief_sha256"),
            model_provenance=_model_provenance(raw.get("model_provenance", {}), f"{path}.model_provenance"),
            review=_review(raw.get("review"), finding_id, f"{path}.review"),
            record_schema_version=record_schema_version,
        ))

    records: tuple[Record, ...] = tuple((*evidence_records, *observations, *findings))
    try:
        validate_record_bundle(records)
    except RecordValidationError as exc:
        raise MissionRecordProjectionError("invalid_record_bundle", str(exc)) from exc
    return records
