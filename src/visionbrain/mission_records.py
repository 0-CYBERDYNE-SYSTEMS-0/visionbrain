"""MLX-free released mission records and their canonical JSON codec."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Literal, Mapping, Sequence, TypeAlias

from .mission_contracts import (
    MAX_DECODED_JPEG_BYTES,
    MAX_FINDING_ITEMS,
    MAX_POLYGON_POINTS,
    EvidenceRef,
    GeometryItem,
    InputTransform,
    MissionTool,
    SourceBinding,
    ToolCallRecord,
)

RECORD_SCHEMA_VERSION = 1
FINDING_TEXT_CITATION_SCHEMA_VERSION = 2

ObservationState: TypeAlias = Literal[
    "observed", "held", "stale", "failed", "unavailable", "capture-time-unknown"
]
ObservationOutcome: TypeAlias = Literal[
    "nonempty", "empty", "failed", "not_run", "unavailable"
]
EvidenceAvailability: TypeAlias = Literal["available", "missing", "corrupt", "rolled_off"]
VisualState: TypeAlias = Literal["candidate", "supported", "unresolved"]
ReviewState: TypeAlias = Literal["pending", "accepted", "corrected", "rejected"]
PacketState: TypeAlias = Literal["draft", "approved", "exported", "superseded"]
ModelValue: TypeAlias = str | int | float | bool | None
CaptureTimeQuality: TypeAlias = Literal["unknown", "unverified", "verified"]


class RecordValidationError(ValueError):
    """A record failed a stable, machine-readable contract check."""

    def __init__(self, reason: str, path: str = "") -> None:
        self.reason = reason
        self.path = path
        super().__init__(f"{reason}{': ' + path if path else ''}")


@dataclass(frozen=True)
class Observation:
    observation_id: str
    mission_id: str
    status: ObservationState
    outcome: ObservationOutcome
    source_id: str | None = None
    source_epoch: str | None = None
    frame_id: int | None = None
    observed_frame_id: int | None = None
    width: int | None = None
    height: int | None = None
    evidence_ids: tuple[str, ...] = ()
    items: tuple[GeometryItem, ...] = ()
    tool_results: tuple[ToolCallRecord, ...] = ()
    detector_tool_revision: str | None = None
    prompt_config_revision: str | None = None
    capture_time_ms: int | None = None
    capture_time_provenance: str | None = None
    capture_time_quality: CaptureTimeQuality = "unknown"
    received_at_ms: int | None = None
    client_session_id: str | None = None
    inference_session_id: str | None = None
    archive_session_id: str | None = None
    error_code: str | None = None
    record_schema_version: int = RECORD_SCHEMA_VERSION


@dataclass(frozen=True)
class EvidenceRecord:
    mission_id: str
    evidence: EvidenceRef
    byte_length: int
    availability: EvidenceAvailability
    availability_reason: str | None = None
    source_observation_id: str | None = None
    source_id: str | None = None
    source_epoch: str | None = None
    frame_id: int | None = None
    capture_time_ms: int | None = None
    capture_time_provenance: str | None = None
    capture_time_quality: CaptureTimeQuality = "unknown"
    created_at_ms: int | None = None
    client_session_id: str | None = None
    inference_session_id: str | None = None
    archive_session_id: str | None = None
    closeup_request_id: str | None = None
    brief_version: int | None = None
    brief_sha256: str | None = None
    model_provenance: tuple[tuple[str, ModelValue], ...] = ()
    record_schema_version: int = RECORD_SCHEMA_VERSION


@dataclass(frozen=True)
class LocalizationStatement:
    claim: str
    label: str
    count: int
    positions: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class Localization:
    status: Literal["supported"]
    basis: str
    evidence_id: str
    statements: tuple[LocalizationStatement, ...]


@dataclass(frozen=True)
class Review:
    finding_id: str
    state: ReviewState = "pending"
    actor: str | None = None
    reviewed_at_ms: int | None = None
    mission_revision: int | None = None
    note: str | None = None
    record_schema_version: int = RECORD_SCHEMA_VERSION


@dataclass(frozen=True)
class Finding:
    finding_id: str
    mission_id: str
    claim: str
    claim_type: Literal["localized_object", "text_read", "visual_hypothesis"]
    visual_state: VisualState
    reason: str
    evidence_id: str
    evidence_refs: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    item_refs: tuple[tuple[str, str], ...] = ()
    items: tuple[GeometryItem, ...] = ()
    localization: Localization | None = None
    source_binding: SourceBinding | None = None
    frame_id: int | None = None
    brief_version: int | None = None
    brief_sha256: str | None = None
    model_provenance: tuple[tuple[str, ModelValue], ...] = ()
    review: Review | None = None
    record_schema_version: int = RECORD_SCHEMA_VERSION
    text_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class InspectionPacket:
    packet_id: str
    mission_id: str
    mission_revision: int
    state: PacketState
    finding_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    created_at_ms: int | None = None
    exported_at_ms: int | None = None
    supersedes_packet_id: str | None = None
    record_schema_version: int = RECORD_SCHEMA_VERSION


@dataclass(frozen=True)
class RuntimeCapability:
    runtime_id: str
    supported_record_versions: tuple[int, ...]
    supported_modes: tuple[str, ...] = ()
    supported_tools: tuple[str, ...] = ()
    max_evidence_bytes: int = MAX_DECODED_JPEG_BYTES
    max_polygon_points: int = MAX_POLYGON_POINTS
    record_schema_version: int = RECORD_SCHEMA_VERSION


# MissionTool already is the MLX-free execute(request, context) protocol.
PerceptionAdapter = MissionTool

Record: TypeAlias = Observation | EvidenceRecord | Review | Finding | InspectionPacket | RuntimeCapability

_RECORD_TYPES: dict[str, type[Any]] = {
    "observation": Observation,
    "evidence": EvidenceRecord,
    "review": Review,
    "finding": Finding,
    "inspection_packet": InspectionPacket,
    "runtime_capability": RuntimeCapability,
}
_RECORD_NAMES = {record_type: name for name, record_type in _RECORD_TYPES.items()}
_OBSERVATION_STATES = {"observed", "held", "stale", "failed", "unavailable", "capture-time-unknown"}
_OBSERVATION_OUTCOMES = {"nonempty", "empty", "failed", "not_run", "unavailable"}
_AVAILABILITY = {"available", "missing", "corrupt", "rolled_off"}
_VISUAL_STATES = {"candidate", "supported", "unresolved"}
_REVIEW_STATES = {"pending", "accepted", "corrected", "rejected"}
_PACKET_STATES = {"draft", "approved", "exported", "superseded"}


def _fail(reason: str, path: str = "") -> None:
    raise RecordValidationError(reason, path)


def _choice(value: Any, options: set[str], reason: str, path: str) -> None:
    if not isinstance(value, str) or value not in options:
        _fail(reason, path)


def _text(value: Any, path: str, *, optional: bool = False, limit: int = 512) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        _fail("invalid_text", path)


def _sha256(value: Any, path: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        _fail("invalid_sha256", path)


def _int(
    value: Any,
    path: str,
    *,
    optional: bool = False,
    minimum: int = 0,
    maximum: int | None = None,
) -> None:
    if optional and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail("invalid_integer", path)
    if maximum is not None and value > maximum:
        _fail("out_of_bounds", path)


def _capture_time(value: int | None, provenance: str | None, quality: str, path: str) -> None:
    _int(value, f"{path}.capture_time_ms", optional=True, maximum=2**63 - 1)
    _choice(quality, {"unknown", "unverified", "verified"}, "invalid_capture_time_quality", f"{path}.capture_time_quality")
    if value is None:
        if provenance is not None or quality != "unknown":
            _fail("capture_time_provenance_mismatch", path)
        return
    _text(provenance, f"{path}.capture_time_provenance", limit=128)
    if quality == "unknown":
        _fail("capture_time_provenance_mismatch", path)


def _number(value: Any, path: str, *, minimum: float = 0.0, maximum: float = 1.0) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail("invalid_number", path)
    if not minimum <= float(value) <= maximum:
        _fail("out_of_bounds", path)


def _strings(values: Any, path: str) -> None:
    if not isinstance(values, tuple):
        _fail("invalid_sequence", path)
    seen: set[str] = set()
    for index, value in enumerate(values):
        _text(value, f"{path}[{index}]")
        if value in seen:
            _fail("duplicate_id", f"{path}[{index}]")
        seen.add(value)


def _text_references(values: Any, path: str) -> None:
    if not isinstance(values, tuple):
        _fail("invalid_sequence", path)
    if len(values) > MAX_FINDING_ITEMS:
        _fail("too_many_text_references", path)
    seen: set[str] = set()
    for index, value in enumerate(values):
        _text(value, f"{path}[{index}]", limit=128)
        if value in seen:
            _fail("duplicate_id", f"{path}[{index}]")
        seen.add(value)


def _model_values(values: Any, path: str) -> None:
    if not isinstance(values, tuple):
        _fail("invalid_sequence", path)
    seen: set[str] = set()
    for index, pair in enumerate(values):
        if not isinstance(pair, tuple) or len(pair) != 2:
            _fail("invalid_model_provenance", f"{path}[{index}]")
        key, value = pair
        _text(key, f"{path}[{index}].key", limit=128)
        if key in seen or (value is not None and not isinstance(value, (str, int, float, bool))):
            _fail("invalid_model_provenance", f"{path}[{index}]")
        if isinstance(value, float) and not math.isfinite(value):
            _fail("invalid_model_provenance", f"{path}[{index}]")
        seen.add(key)


def _validate_transform(value: InputTransform, path: str) -> None:
    for name in ("origin_width", "origin_height", "width", "height"):
        _int(getattr(value, name), f"{path}.{name}", optional=True, minimum=1)
    if value.rotation_degrees not in (0, 90, 180, 270):
        _fail("invalid_transform", f"{path}.rotation_degrees")
    if not isinstance(value.resized, bool) or not isinstance(value.flipped, bool):
        _fail("invalid_transform", path)
    for name in ("scale_x", "scale_y"):
        scale = getattr(value, name)
        if scale is not None:
            _number(scale, f"{path}.{name}", minimum=0.0000000001, maximum=math.inf)


def _validate_geometry(item: GeometryItem, path: str) -> None:
    _text(item.item_id, f"{path}.item_id", limit=128)
    _text(item.label, f"{path}.label", limit=128)
    _text(item.source, f"{path}.source", limit=128)
    _number(item.score, f"{path}.score")
    if not isinstance(item.box, tuple) or len(item.box) != 4:
        _fail("invalid_geometry", f"{path}.box")
    for index, coordinate in enumerate(item.box):
        _number(coordinate, f"{path}.box[{index}]")
    if item.box[2] <= item.box[0] or item.box[3] <= item.box[1]:
        _fail("degenerate_geometry", f"{path}.box")
    if item.polygon is not None:
        if not isinstance(item.polygon, tuple) or not 3 <= len(item.polygon) <= MAX_POLYGON_POINTS:
            _fail("invalid_polygon", f"{path}.polygon")
        for index, point in enumerate(item.polygon):
            if not isinstance(point, tuple) or len(point) != 2:
                _fail("invalid_polygon", f"{path}.polygon[{index}]")
            for coordinate_index, coordinate in enumerate(point):
                _number(coordinate, f"{path}.polygon[{index}][{coordinate_index}]")
        area = sum(
            item.polygon[index][0] * item.polygon[(index + 1) % len(item.polygon)][1]
            - item.polygon[(index + 1) % len(item.polygon)][0] * item.polygon[index][1]
            for index in range(len(item.polygon))
        )
        if area == 0:
            _fail("degenerate_geometry", f"{path}.polygon")


def _validate_scalar_mapping(value: Any, path: str) -> None:
    if not isinstance(value, Mapping):
        _fail("invalid_mapping", path)
    for key, item in value.items():
        _text(key, f"{path}.key", limit=128)
        if item is not None and not isinstance(item, (str, int, float, bool)):
            _fail("invalid_mapping_value", f"{path}.{key}")
        if isinstance(item, float) and not math.isfinite(item):
            _fail("invalid_number", f"{path}.{key}")


def _validate_tool_call(value: ToolCallRecord, path: str) -> None:
    if not isinstance(value, ToolCallRecord):
        _fail("invalid_tool_result", path)
    _text(value.tool_result_id, f"{path}.tool_result_id", limit=128)
    _text(value.tool, f"{path}.tool", limit=128)
    _choice(value.status, {"ok", "empty", "unsupported", "failed", "timeout"}, "invalid_tool_status", f"{path}.status")
    _text(value.input_evidence_id, f"{path}.input_evidence_id", limit=128)
    if not isinstance(value.items, tuple):
        _fail("invalid_sequence", f"{path}.items")
    for index, item in enumerate(value.items):
        _validate_geometry(item, f"{path}.items[{index}]")
    _strings(value.evidence_ids, f"{path}.evidence_ids")
    _strings(value.unavailable_evidence_ids, f"{path}.unavailable_evidence_ids")
    if not isinstance(value.text, str) or len(value.text) > 4_000:
        _fail("invalid_text", f"{path}.text")
    if value.status == "empty" and (value.items or value.evidence_ids or value.text.strip()):
        _fail("tool_status_output_mismatch", f"{path}.status")
    _text(value.error_code, f"{path}.error_code", optional=True, limit=128)
    _int(value.brief_version, f"{path}.brief_version", minimum=1)
    _sha256(value.brief_sha256, f"{path}.brief_sha256", optional=True)
    _validate_scalar_mapping(value.model_provenance, f"{path}.model_provenance")
    if value.source_binding is not None:
        if not isinstance(value.source_binding, Mapping):
            _fail("invalid_source_binding", f"{path}.source_binding")
        if set(value.source_binding) != {"source_id", "source_epoch"}:
            _fail("invalid_source_binding", f"{path}.source_binding")
        _text(value.source_binding["source_id"], f"{path}.source_binding.source_id", limit=128)
        _text(value.source_binding["source_epoch"], f"{path}.source_binding.source_epoch", limit=128)
    _int(value.frame_id, f"{path}.frame_id", optional=True)
    _sha256(value.input_sha256, f"{path}.input_sha256", optional=True)
    _validate_scalar_mapping(value.evidence_sha256, f"{path}.evidence_sha256")
    for evidence_id, digest in value.evidence_sha256.items():
        _text(evidence_id, f"{path}.evidence_sha256.key", limit=128)
        _sha256(digest, f"{path}.evidence_sha256.{evidence_id}")
    _int(value.created_at_ms, f"{path}.created_at_ms", optional=True, maximum=2**63 - 1)


def validate_record(record: Record) -> Record:
    """Validate one immutable released record, returning it unchanged."""
    if type(record) not in _RECORD_NAMES:
        _fail("unsupported_record_type")
    _int(record.record_schema_version, "record_schema_version", minimum=1)
    supported_versions = (
        {RECORD_SCHEMA_VERSION, FINDING_TEXT_CITATION_SCHEMA_VERSION}
        if type(record) is Finding
        else {RECORD_SCHEMA_VERSION}
    )
    if record.record_schema_version not in supported_versions:
        _fail("unsupported_record_version", "record_schema_version")

    if isinstance(record, Observation):
        _text(record.observation_id, "observation_id", limit=128)
        _text(record.mission_id, "mission_id", limit=128)
        _choice(record.status, _OBSERVATION_STATES, "invalid_observation_state", "status")
        _choice(record.outcome, _OBSERVATION_OUTCOMES, "invalid_observation_outcome", "outcome")
        _text(record.source_id, "source_id", optional=True, limit=128)
        _text(record.source_epoch, "source_epoch", optional=True, limit=128)
        if (record.source_id is None) != (record.source_epoch is None):
            _fail("incomplete_source_binding", "source_epoch")
        for name in ("frame_id", "observed_frame_id"):
            _int(getattr(record, name), name, optional=True)
        _int(record.received_at_ms, "received_at_ms", optional=True, maximum=2**63 - 1)
        if record.status == "held" and record.observed_frame_id is None:
            _fail("held_observation_missing_observed_frame", "observed_frame_id")
        if record.status == "failed" and (record.outcome != "failed" or not record.error_code):
            _fail("failed_observation_missing_error", "error_code")
        if record.outcome == "failed" and record.status != "failed":
            _fail("observation_status_outcome_mismatch", "outcome")
        if record.status == "unavailable" and record.outcome not in {"unavailable", "not_run"}:
            _fail("observation_status_outcome_mismatch", "outcome")
        if record.outcome in {"unavailable", "not_run"} and record.status != "unavailable":
            _fail("observation_status_outcome_mismatch", "outcome")
        _capture_time(
            record.capture_time_ms,
            record.capture_time_provenance,
            record.capture_time_quality,
            "observation",
        )
        if record.status == "capture-time-unknown" and record.capture_time_quality == "verified":
            _fail("capture_time_state_mismatch", "capture_time_quality")
        _text(record.error_code, "error_code", optional=True, limit=128)
        _strings(record.evidence_ids, "evidence_ids")
        if not isinstance(record.items, tuple):
            _fail("invalid_sequence", "items")
        for index, item in enumerate(record.items):
            _validate_geometry(item, f"items[{index}]")
        if not isinstance(record.tool_results, tuple):
            _fail("invalid_sequence", "tool_results")
        tool_ids: set[str] = set()
        for index, result in enumerate(record.tool_results):
            _validate_tool_call(result, f"tool_results[{index}]")
            if result.tool_result_id in tool_ids:
                _fail("duplicate_id", f"tool_results[{index}].tool_result_id")
            tool_ids.add(result.tool_result_id)
        if record.outcome == "empty" and (
            record.items
            or any(
                result.status == "ok"
                and (result.items or result.evidence_ids or result.text.strip())
                for result in record.tool_results
            )
        ):
            _fail("observation_empty_has_output", "outcome")
        _text(record.detector_tool_revision, "detector_tool_revision", optional=True, limit=128)
        _text(record.prompt_config_revision, "prompt_config_revision", optional=True, limit=128)
        for name in ("client_session_id", "inference_session_id", "archive_session_id"):
            _text(getattr(record, name), name, optional=True, limit=128)
        if (record.width is None) != (record.height is None):
            _fail("incomplete_dimensions", "width")
        _int(record.width, "width", optional=True, minimum=1)
        _int(record.height, "height", optional=True, minimum=1)

    elif isinstance(record, EvidenceRecord):
        _text(record.mission_id, "mission_id", limit=128)
        ref = record.evidence
        _text(ref.evidence_id, "evidence.evidence_id", limit=128)
        if not isinstance(ref.sha256, str) or len(ref.sha256) != 64 or any(c not in "0123456789abcdef" for c in ref.sha256):
            _fail("invalid_sha256", "evidence.sha256")
        _int(ref.width, "evidence.width", minimum=1)
        _int(ref.height, "evidence.height", minimum=1)
        _int(record.byte_length, "byte_length", minimum=1)
        _text(ref.kind, "evidence.kind", limit=64)
        _text(ref.parent_evidence_id, "evidence.parent_evidence_id", optional=True, limit=128)
        if ref.crop_box is not None:
            if not isinstance(ref.crop_box, tuple) or len(ref.crop_box) != 4:
                _fail("invalid_geometry", "evidence.crop_box")
            for index, coordinate in enumerate(ref.crop_box):
                _number(coordinate, f"evidence.crop_box[{index}]")
            if ref.crop_box[2] <= ref.crop_box[0] or ref.crop_box[3] <= ref.crop_box[1]:
                _fail("degenerate_geometry", "evidence.crop_box")
        if ref.input_transform is not None:
            _validate_transform(ref.input_transform, "evidence.input_transform")
        _choice(record.availability, _AVAILABILITY, "invalid_evidence_availability", "availability")
        if record.availability == "available":
            if record.availability_reason is not None:
                _fail("availability_reason_mismatch", "availability_reason")
        else:
            _text(record.availability_reason, "availability_reason", limit=128)
        _text(record.source_observation_id, "source_observation_id", optional=True, limit=128)
        _text(record.source_id, "source_id", optional=True, limit=128)
        _text(record.source_epoch, "source_epoch", optional=True, limit=128)
        if (record.source_id is None) != (record.source_epoch is None):
            _fail("incomplete_source_binding", "source_epoch")
        for name in ("frame_id", "created_at_ms", "brief_version"):
            _int(getattr(record, name), name, optional=True, minimum=1 if name == "brief_version" else 0)
        _capture_time(
            record.capture_time_ms,
            record.capture_time_provenance,
            record.capture_time_quality,
            "evidence",
        )
        _int(record.created_at_ms, "created_at_ms", optional=True, maximum=2**63 - 1)
        for name in ("client_session_id", "inference_session_id", "archive_session_id", "closeup_request_id"):
            _text(getattr(record, name), name, optional=True, limit=128)
        _sha256(record.brief_sha256, "brief_sha256", optional=True)
        _model_values(record.model_provenance, "model_provenance")

    elif isinstance(record, Review):
        _text(record.finding_id, "finding_id", limit=128)
        _choice(record.state, _REVIEW_STATES, "invalid_review_state", "state")
        if record.state == "pending":
            if any(value is not None for value in (record.actor, record.reviewed_at_ms, record.mission_revision)):
                _fail("pending_review_has_attribution", "state")
        else:
            _text(record.actor, "actor", limit=128)
            _int(record.reviewed_at_ms, "reviewed_at_ms", minimum=1)
            _int(record.mission_revision, "mission_revision", minimum=1)
        _text(record.note, "note", optional=True, limit=1_000)

    elif isinstance(record, Finding):
        _text(record.finding_id, "finding_id", limit=128)
        _text(record.mission_id, "mission_id", limit=128)
        _text(record.claim, "claim", limit=500)
        _choice(record.claim_type, {"localized_object", "text_read", "visual_hypothesis"}, "invalid_claim_type", "claim_type")
        _choice(record.visual_state, _VISUAL_STATES, "invalid_visual_state", "visual_state")
        _text(record.reason, "reason", limit=256)
        _text(record.evidence_id, "evidence_id", limit=128)
        _strings(record.evidence_refs, "evidence_refs")
        _strings(record.observation_ids, "observation_ids")
        if not isinstance(record.text_refs, tuple):
            _fail("invalid_sequence", "text_refs")
        if record.record_schema_version == RECORD_SCHEMA_VERSION:
            if record.text_refs:
                _fail("text_refs_require_record_version_2", "text_refs")
        else:
            _text_references(record.text_refs, "text_refs")
            if record.claim_type == "text_read" and not record.text_refs:
                _fail("missing_text_reference", "text_refs")
            if record.claim_type != "text_read" and record.text_refs:
                _fail("unexpected_text_reference", "text_refs")
        for index, pair in enumerate(record.item_refs):
            if not isinstance(pair, tuple) or len(pair) != 2:
                _fail("invalid_item_reference", f"item_refs[{index}]")
            _text(pair[0], f"item_refs[{index}].tool_result_id", limit=128)
            _text(pair[1], f"item_refs[{index}].item_id", limit=128)
        for index, item in enumerate(record.items):
            _validate_geometry(item, f"items[{index}]")
        if record.source_binding is not None:
            _text(record.source_binding.source_id, "source_binding.source_id", limit=128)
            _text(record.source_binding.source_epoch, "source_binding.source_epoch", limit=128)
        _int(record.frame_id, "frame_id", optional=True)
        _int(record.brief_version, "brief_version", optional=True, minimum=1)
        _sha256(record.brief_sha256, "brief_sha256", optional=True)
        _model_values(record.model_provenance, "model_provenance")
        if record.review is not None:
            validate_record(record.review)
            if record.review.finding_id != record.finding_id:
                _fail("review_finding_mismatch", "review.finding_id")
        if record.localization is not None:
            _validate_localization(record.localization)

    elif isinstance(record, InspectionPacket):
        _text(record.packet_id, "packet_id", limit=128)
        _text(record.mission_id, "mission_id", limit=128)
        _int(record.mission_revision, "mission_revision", minimum=1)
        _choice(record.state, _PACKET_STATES, "invalid_packet_state", "state")
        for name in ("finding_ids", "observation_ids", "evidence_ids"):
            _strings(getattr(record, name), name)
        _int(record.created_at_ms, "created_at_ms", optional=True)
        _int(record.exported_at_ms, "exported_at_ms", optional=True)
        _text(record.supersedes_packet_id, "supersedes_packet_id", optional=True, limit=128)

    elif isinstance(record, RuntimeCapability):
        _text(record.runtime_id, "runtime_id", limit=128)
        _ints(record.supported_record_versions, "supported_record_versions", minimum=1)
        _strings(record.supported_modes, "supported_modes")
        _strings(record.supported_tools, "supported_tools")
        _int(record.max_evidence_bytes, "max_evidence_bytes", minimum=1)
        _int(record.max_polygon_points, "max_polygon_points", minimum=3)
        if record.max_evidence_bytes > MAX_DECODED_JPEG_BYTES or record.max_polygon_points > MAX_POLYGON_POINTS:
            _fail("unsupported_capability_limit", "runtime_capability")
    return record


def _ints(values: Any, path: str, *, minimum: int) -> None:
    if not isinstance(values, tuple):
        _fail("invalid_sequence", path)
    seen: set[int] = set()
    for index, value in enumerate(values):
        _int(value, f"{path}[{index}]", minimum=minimum)
        if value in seen:
            _fail("duplicate_value", f"{path}[{index}]")
        seen.add(value)


def _validate_localization(value: Localization) -> None:
    _choice(value.status, {"supported"}, "invalid_localization_state", "localization.status")
    _text(value.basis, "localization.basis", limit=128)
    _text(value.evidence_id, "localization.evidence_id", limit=128)
    if not isinstance(value.statements, tuple):
        _fail("invalid_sequence", "localization.statements")
    for index, statement in enumerate(value.statements):
        _text(statement.claim, f"localization.statements[{index}].claim", limit=500)
        _text(statement.label, f"localization.statements[{index}].label", limit=128)
        _int(statement.count, f"localization.statements[{index}].count", minimum=1)
        if len(statement.positions) != statement.count:
            _fail("localization_count_mismatch", f"localization.statements[{index}].positions")
        for position_index, position in enumerate(statement.positions):
            if not isinstance(position, tuple) or len(position) != 2:
                _fail("invalid_geometry", f"localization.statements[{index}].positions[{position_index}]")
            for coordinate_index, coordinate in enumerate(position):
                _number(coordinate, f"localization.statements[{index}].positions[{position_index}][{coordinate_index}]")


def _capability_shape(value: RuntimeCapability) -> tuple[Any, ...]:
    return (
        value.supported_record_versions,
        value.supported_modes,
        value.supported_tools,
        value.max_evidence_bytes,
        value.max_polygon_points,
    )


def _record_geometry(record: Record) -> tuple[GeometryItem, ...]:
    if isinstance(record, Observation):
        return record.items + tuple(item for result in record.tool_results for item in result.items)
    if isinstance(record, Finding):
        return record.items
    return ()


def _plain(value: Any) -> Any:
    if type(value) in _RECORD_NAMES:
        return {"record_type": _RECORD_NAMES[type(value)], **{field.name: _plain(getattr(value, field.name)) for field in fields(value)}}
    if is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    _fail("unsupported_json_value")


def record_to_dict(record: Record) -> dict[str, Any]:
    """Return the strict JSON-compatible representation of one record."""
    validate_record(record)
    value = _plain(record)
    if isinstance(record, Finding) and record.record_schema_version == RECORD_SCHEMA_VERSION:
        value.pop("text_refs")
    return value


def serialize_record(record: Record) -> str:
    """Serialize a record as canonical sorted, compact JSON."""
    return json.dumps(record_to_dict(record), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _object(value: Any, expected: set[str], path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("invalid_object", path)
    keys = set(value)
    if keys - expected:
        _fail("unknown_field", path)
    if expected - keys:
        _fail("missing_field", path)
    return value


def _evidence_ref_from_dict(value: Any) -> EvidenceRef:
    names = {field.name for field in fields(EvidenceRef)}
    data = dict(_object(value, names, "evidence"))
    if data["crop_box"] is not None:
        if not isinstance(data["crop_box"], list):
            _fail("invalid_geometry", "evidence.crop_box")
        data["crop_box"] = tuple(data["crop_box"])
    if data["input_transform"] is not None:
        transform_names = {field.name for field in fields(InputTransform)}
        transform_data = _object(data["input_transform"], transform_names, "evidence.input_transform")
        try:
            data["input_transform"] = InputTransform(**transform_data)
        except (TypeError, ValueError):
            _fail("invalid_transform", "evidence.input_transform")
    try:
        return EvidenceRef(**data)
    except (TypeError, ValueError):
        _fail("invalid_nested_record", "evidence")


def _geometry_item_from_dict(value: Any, path: str) -> GeometryItem:
    names = {field.name for field in fields(GeometryItem)}
    data = dict(_object(value, names, path))
    if not isinstance(data["box"], list):
        _fail("invalid_geometry", f"{path}.box")
    data["box"] = tuple(data["box"])
    if data["polygon"] is not None:
        if not isinstance(data["polygon"], list):
            _fail("invalid_polygon", f"{path}.polygon")
        points = []
        for index, point in enumerate(data["polygon"]):
            if not isinstance(point, list):
                _fail("invalid_polygon", f"{path}.polygon[{index}]")
            points.append(tuple(point))
        data["polygon"] = tuple(points)
    try:
        return GeometryItem(**data)
    except (TypeError, ValueError):
        _fail("invalid_nested_record", path)


def _source_binding_from_dict(value: Any) -> SourceBinding:
    names = {field.name for field in fields(SourceBinding)}
    data = _object(value, names, "source_binding")
    try:
        return SourceBinding(**data)
    except (TypeError, ValueError):
        _fail("invalid_nested_record", "source_binding")


def _tool_call_from_dict(value: Any, path: str) -> ToolCallRecord:
    names = {field.name for field in fields(ToolCallRecord)}
    data = dict(_object(value, names, path))
    for name in ("items", "evidence_ids", "unavailable_evidence_ids"):
        if not isinstance(data[name], list):
            _fail("invalid_sequence", f"{path}.{name}")
    data["items"] = tuple(
        _geometry_item_from_dict(item, f"{path}.items[{index}]")
        for index, item in enumerate(data["items"])
    )
    data["evidence_ids"] = tuple(data["evidence_ids"])
    data["unavailable_evidence_ids"] = tuple(data["unavailable_evidence_ids"])
    for name in ("model_provenance", "evidence_sha256"):
        if not isinstance(data[name], Mapping):
            _fail("invalid_mapping", f"{path}.{name}")
        data[name] = dict(data[name])
    if data["source_binding"] is not None:
        if not isinstance(data["source_binding"], Mapping):
            _fail("invalid_source_binding", f"{path}.source_binding")
        data["source_binding"] = dict(data["source_binding"])
    try:
        return ToolCallRecord(**data)
    except (TypeError, ValueError):
        _fail("invalid_tool_result", path)


def _tuple_field(data: Mapping[str, Any], key: str, path: str) -> tuple[Any, ...]:
    value = data[key]
    if not isinstance(value, list):
        _fail("invalid_sequence", path)
    return tuple(value)


def _model_tuple(data: Mapping[str, Any], key: str, path: str) -> tuple[tuple[str, ModelValue], ...]:
    pairs = _tuple_field(data, key, path)
    output = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, list) or len(pair) != 2:
            _fail("invalid_model_provenance", f"{path}[{index}]")
        output.append((pair[0], pair[1]))
    return tuple(output)


def _review_from_dict(value: Any) -> Review | None:
    if value is None:
        return None
    parsed = record_from_dict(value)
    if not isinstance(parsed, Review):
        _fail("invalid_nested_record", "review")
    return parsed


def _localization_from_dict(value: Any) -> Localization | None:
    if value is None:
        return None
    names = {field.name for field in fields(Localization)}
    data = _object(value, names, "localization")
    raw_statements = data["statements"]
    if not isinstance(raw_statements, list):
        _fail("invalid_sequence", "localization.statements")
    statements = []
    for index, raw in enumerate(raw_statements):
        statement_names = {field.name for field in fields(LocalizationStatement)}
        item = _object(raw, statement_names, f"localization.statements[{index}]")
        positions = item["positions"]
        if not isinstance(positions, list):
            _fail("invalid_sequence", f"localization.statements[{index}].positions")
        statements.append(LocalizationStatement(
            claim=item["claim"], label=item["label"], count=item["count"],
            positions=tuple(tuple(position) if isinstance(position, list) else position for position in positions),
        ))
    return Localization(data["status"], data["basis"], data["evidence_id"], tuple(statements))


def record_from_dict(value: Any) -> Record:
    """Decode a record dictionary, rejecting unknown, missing, or malformed fields."""
    if not isinstance(value, Mapping):
        _fail("invalid_object")
    name = value.get("record_type")
    cls = _RECORD_TYPES.get(name) if isinstance(name, str) else None
    if cls is None:
        _fail("unknown_record_type", "record_type")
    expected = {field.name for field in fields(cls)} | {"record_type"}
    if cls is Finding and value.get("record_schema_version") != FINDING_TEXT_CITATION_SCHEMA_VERSION:
        expected.discard("text_refs")
    data = _object(value, expected, "record")
    values = dict(data)
    values.pop("record_type")
    if cls is EvidenceRecord:
        values["evidence"] = _evidence_ref_from_dict(values["evidence"])
        values["model_provenance"] = _model_tuple(values, "model_provenance", "model_provenance")
    elif cls is Observation:
        values["evidence_ids"] = _tuple_field(values, "evidence_ids", "evidence_ids")
        raw_items = _tuple_field(values, "items", "items")
        values["items"] = tuple(
            _geometry_item_from_dict(item, f"items[{index}]")
            for index, item in enumerate(raw_items)
        )
        raw_results = _tuple_field(values, "tool_results", "tool_results")
        values["tool_results"] = tuple(
            _tool_call_from_dict(item, f"tool_results[{index}]")
            for index, item in enumerate(raw_results)
        )
    elif cls is Finding:
        values["evidence_refs"] = _tuple_field(values, "evidence_refs", "evidence_refs")
        values["observation_ids"] = _tuple_field(values, "observation_ids", "observation_ids")
        if "text_refs" in values:
            values["text_refs"] = _tuple_field(values, "text_refs", "text_refs")
        values["item_refs"] = tuple(tuple(pair) if isinstance(pair, list) else pair for pair in _tuple_field(values, "item_refs", "item_refs"))
        raw_items = _tuple_field(values, "items", "items")
        values["items"] = tuple(_geometry_item_from_dict(item, f"items[{index}]") for index, item in enumerate(raw_items))
        if values["source_binding"] is not None:
            values["source_binding"] = _source_binding_from_dict(values["source_binding"])
        values["localization"] = _localization_from_dict(values["localization"])
        values["review"] = _review_from_dict(values["review"])
        values["model_provenance"] = _model_tuple(values, "model_provenance", "model_provenance")
    elif cls is InspectionPacket:
        for key in ("finding_ids", "observation_ids", "evidence_ids"):
            values[key] = _tuple_field(values, key, key)
    elif cls is RuntimeCapability:
        values["supported_record_versions"] = _tuple_field(values, "supported_record_versions", "supported_record_versions")
        values["supported_modes"] = _tuple_field(values, "supported_modes", "supported_modes")
        values["supported_tools"] = _tuple_field(values, "supported_tools", "supported_tools")
    try:
        record = cls(**values)
    except (TypeError, ValueError):
        _fail("invalid_record_fields")
    return validate_record(record)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", key)
        result[key] = value
    return result


def _reject_constant(_value: str) -> None:
    _fail("invalid_number")


def deserialize_record(payload: str | bytes) -> Record:
    """Parse canonical JSON and strictly validate its record shape and values."""
    try:
        value = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except RecordValidationError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        _fail("invalid_json")
    return record_from_dict(value)


def validate_record_bundle(
    records: Sequence[Record], capability: RuntimeCapability | None = None
) -> None:
    """Check typed IDs, references, and local evidence/source lineage only."""
    ids: dict[type[Any], set[str]] = {}
    observations: dict[str, Observation] = {}
    evidence: dict[str, EvidenceRecord] = {}
    findings: dict[str, Finding] = {}
    packets: dict[str, InspectionPacket] = {}
    capabilities: list[RuntimeCapability] = []
    reviews: list[Review] = []
    for record in records:
        validate_record(record)
        if isinstance(record, Observation):
            key, identifier = Observation, record.observation_id
            observations[identifier] = record
        elif isinstance(record, EvidenceRecord):
            key, identifier = EvidenceRecord, record.evidence.evidence_id
            evidence[identifier] = record
        elif isinstance(record, Finding):
            key, identifier = Finding, record.finding_id
            findings[identifier] = record
        elif isinstance(record, InspectionPacket):
            key, identifier = InspectionPacket, record.packet_id
            packets[identifier] = record
        elif isinstance(record, Review):
            reviews.append(record)
            key, identifier = Review, record.finding_id
        elif isinstance(record, RuntimeCapability):
            capabilities.append(record)
            key, identifier = RuntimeCapability, record.runtime_id
        else:
            _fail("unsupported_record_type")
        if identifier in ids.setdefault(key, set()):
            _fail("duplicate_id", identifier)
        ids[key].add(identifier)
    if len({_capability_shape(item) for item in capabilities}) > 1:
        _fail("conflicting_capabilities", "runtime_capability")
    if capability is not None and capabilities and capability not in capabilities:
        _fail("capability_mismatch", "runtime_capability")
    effective_capability = capability or (capabilities[0] if capabilities else None)
    if effective_capability is not None:
        validate_record(effective_capability)
        for record in records:
            if record.record_schema_version not in effective_capability.supported_record_versions:
                _fail("capability_mismatch", "record_schema_version")
        for record in records:
            for item in _record_geometry(record):
                if item.polygon is not None and len(item.polygon) > effective_capability.max_polygon_points:
                    _fail("capability_mismatch", "max_polygon_points")
        for item in evidence.values():
            if item.byte_length > effective_capability.max_evidence_bytes:
                _fail("capability_mismatch", "max_evidence_bytes")

    tool_outputs: dict[str, tuple[Observation, ToolCallRecord]] = {}
    for observation in observations.values():
        for evidence_id in observation.evidence_ids:
            if evidence_id not in evidence:
                _fail("missing_reference", f"observation.evidence_ids:{evidence_id}")
            if evidence[evidence_id].mission_id != observation.mission_id:
                _fail("observation_evidence_mission_mismatch", f"observation:{observation.observation_id}")
        for result in observation.tool_results:
            if result.tool_result_id in tool_outputs:
                _fail("duplicate_id", f"tool_result:{result.tool_result_id}")
            tool_outputs[result.tool_result_id] = (observation, result)
            referenced = {result.input_evidence_id, *result.evidence_ids, *result.unavailable_evidence_ids}
            if not referenced.issubset(evidence):
                _fail("missing_reference", f"observation.tool_results:{result.tool_result_id}")
            if any(evidence[value].mission_id != observation.mission_id for value in referenced):
                _fail("observation_evidence_mission_mismatch", f"observation:{observation.observation_id}")
            if result.input_sha256 is not None and result.input_sha256 != evidence[result.input_evidence_id].evidence.sha256:
                _fail("tool_input_hash_mismatch", f"tool_result:{result.tool_result_id}")
            if not set(result.evidence_sha256).issubset(result.evidence_ids):
                _fail("tool_evidence_reference_mismatch", f"tool_result:{result.tool_result_id}")
            for evidence_id, digest in result.evidence_sha256.items():
                if digest != evidence[evidence_id].evidence.sha256:
                    _fail("tool_evidence_hash_mismatch", f"tool_result:{result.tool_result_id}")
            if result.source_binding is not None and observation.source_id is not None:
                if (
                    result.source_binding["source_id"] != observation.source_id
                    or result.source_binding["source_epoch"] != observation.source_epoch
                ):
                    _fail("tool_observation_lineage_mismatch", f"tool_result:{result.tool_result_id}")
            expected_frame_id = observation.observed_frame_id if observation.status == "held" else observation.frame_id
            if result.frame_id is not None and expected_frame_id is not None and result.frame_id != expected_frame_id:
                _fail("tool_observation_lineage_mismatch", f"tool_result:{result.tool_result_id}")
    for evidence_id, item in evidence.items():
        if item.source_observation_id is not None:
            observation = observations.get(item.source_observation_id)
            if observation is None:
                _fail("missing_reference", f"evidence.source_observation_id:{item.source_observation_id}")
            if item.mission_id != observation.mission_id:
                _fail("evidence_observation_mission_mismatch", f"evidence:{evidence_id}")
            expected_frame_id = observation.observed_frame_id if observation.status == "held" else observation.frame_id
            if (
                (item.source_id is not None and item.source_id != observation.source_id)
                or (item.source_epoch is not None and item.source_epoch != observation.source_epoch)
                or (observation.status == "held" and item.frame_id != expected_frame_id)
                or (
                    observation.status != "held"
                    and item.frame_id is not None
                    and item.frame_id != expected_frame_id
                )
            ):
                _fail("evidence_observation_lineage_mismatch", f"evidence:{evidence_id}")
        if item.evidence.parent_evidence_id is not None and item.evidence.parent_evidence_id not in evidence:
            _fail("missing_reference", f"evidence.parent_evidence_id:{item.evidence.parent_evidence_id}")
        if item.evidence.parent_evidence_id is not None:
            parent = evidence[item.evidence.parent_evidence_id]
            if parent.mission_id != item.mission_id:
                _fail("parent_evidence_mission_mismatch", f"evidence:{evidence_id}")
    for finding in findings.values():
        evidence_refs = set(finding.evidence_refs) | {finding.evidence_id}
        if not evidence_refs.issubset(evidence):
            _fail("missing_reference", "finding.evidence_refs")
        if not set(finding.observation_ids).issubset(observations):
            _fail("missing_reference", "finding.observation_ids")
        if any(evidence[value].mission_id != finding.mission_id for value in evidence_refs):
            _fail("finding_evidence_mission_mismatch", f"finding:{finding.finding_id}")
        if any(observations[value].mission_id != finding.mission_id for value in finding.observation_ids):
            _fail("finding_observation_mission_mismatch", f"finding:{finding.finding_id}")
        if finding.source_binding is not None:
            for value in finding.observation_ids:
                observation = observations[value]
                if observation.source_id is not None and (
                    finding.source_binding.source_id != observation.source_id
                    or finding.source_binding.source_epoch != observation.source_epoch
                ):
                    _fail("finding_observation_lineage_mismatch", f"finding:{finding.finding_id}")
        for value in evidence_refs:
            evidence_record = evidence[value]
            if (
                finding.source_binding is not None
                and evidence_record.source_id is not None
                and (
                    finding.source_binding.source_id != evidence_record.source_id
                    or finding.source_binding.source_epoch != evidence_record.source_epoch
                )
            ):
                _fail("finding_evidence_lineage_mismatch", f"finding:{finding.finding_id}")
            if (
                evidence_record.source_observation_id is not None
                and finding.observation_ids
                and evidence_record.source_observation_id not in finding.observation_ids
            ):
                _fail("finding_evidence_lineage_mismatch", f"finding:{finding.finding_id}")
        for tool_result_id, item_id in finding.item_refs:
            tool_entry = tool_outputs.get(tool_result_id)
            if tool_entry is None or item_id not in {item.item_id for item in tool_entry[1].items}:
                _fail("missing_tool_item_reference", f"finding:{finding.finding_id}")
            observation, _result = tool_entry
            if observation.mission_id != finding.mission_id:
                _fail("finding_tool_mission_mismatch", f"finding:{finding.finding_id}")
            if finding.observation_ids and observation.observation_id not in finding.observation_ids:
                _fail("finding_tool_observation_mismatch", f"finding:{finding.finding_id}")
            if finding.source_binding is not None and observation.source_id is not None and (
                finding.source_binding.source_id != observation.source_id
                or finding.source_binding.source_epoch != observation.source_epoch
            ):
                _fail("finding_tool_lineage_mismatch", f"finding:{finding.finding_id}")
        for tool_result_id in finding.text_refs:
            tool_entry = tool_outputs.get(tool_result_id)
            if tool_entry is None:
                _fail("missing_reference", f"finding.text_refs:{tool_result_id}")
            observation, result = tool_entry
            if observation.mission_id != finding.mission_id:
                _fail("finding_tool_mission_mismatch", f"finding:{finding.finding_id}")
            if observation.observation_id not in finding.observation_ids:
                _fail("finding_tool_observation_mismatch", f"finding:{finding.finding_id}")
            if finding.source_binding is not None and observation.source_id is not None and (
                finding.source_binding.source_id != observation.source_id
                or finding.source_binding.source_epoch != observation.source_epoch
            ):
                _fail("finding_tool_lineage_mismatch", f"finding:{finding.finding_id}")
            if result.tool != "read_text":
                _fail("finding_text_tool_mismatch", f"finding:{finding.finding_id}")
            if result.status != "ok":
                _fail("finding_text_tool_status_mismatch", f"finding:{finding.finding_id}")
        if finding.review is not None and finding.review.finding_id != finding.finding_id:
            _fail("review_finding_mismatch", "finding.review")
        if finding.localization is not None and finding.localization.evidence_id not in evidence_refs:
            _fail("finding_localization_evidence_mismatch", f"finding:{finding.finding_id}")
    for review in reviews:
        if review.finding_id not in findings:
            _fail("missing_reference", f"review.finding_id:{review.finding_id}")
    for packet in packets.values():
        packet_findings = [findings.get(value) for value in packet.finding_ids]
        packet_observations = [observations.get(value) for value in packet.observation_ids]
        if any(item is None for item in packet_findings + packet_observations):
            _fail("missing_reference", f"packet:{packet.packet_id}")
        if any(item.mission_id != packet.mission_id for item in packet_findings + packet_observations if item is not None):
            _fail("packet_mission_mismatch", f"packet:{packet.packet_id}")
        if not set(packet.evidence_ids).issubset(evidence):
            _fail("missing_reference", f"packet.evidence_ids:{packet.packet_id}")
        if any(evidence[value].mission_id != packet.mission_id for value in packet.evidence_ids):
            _fail("packet_mission_mismatch", f"packet:{packet.packet_id}")
        if packet.supersedes_packet_id is not None:
            if packet.supersedes_packet_id not in packets:
                _fail("missing_reference", f"packet.supersedes_packet_id:{packet.packet_id}")
            if packets[packet.supersedes_packet_id].mission_id != packet.mission_id:
                _fail("packet_supersedes_mission_mismatch", f"packet:{packet.packet_id}")
