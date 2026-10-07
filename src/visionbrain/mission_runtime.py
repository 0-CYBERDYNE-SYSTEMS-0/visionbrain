"""Canonical bounded mission authority and orchestration loop.

This module remains import-safe on hosts without MLX. Native inference is
injected, performed one non-preemptible operation at a time, and never runs in
the WebSocket request handler.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import math
import time
import uuid
from collections.abc import Collection, Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from .mission_contracts import (
    MAX_ACTIVE_SECONDS,
    MAX_BRIEF_VERSIONS,
    MAX_BRIEF_FULL_TEXT_VERSIONS,
    MAX_DECODED_JPEG_BYTES,
    MAX_EXPERTISE_CHARS,
    MAX_FINDINGS,
    MAX_FINDING_ITEMS,
    MAX_GENERATIONS_PER_CYCLE,
    MAX_GENERATIONS_PER_WINDOW,
    MAX_GOAL_CHARS,
    MAX_HUMAN_WAIT_SECONDS,
    MAX_MISSION_EVIDENCE,
    MAX_MISSION_EVIDENCE_HISTORY,
    MAX_MISSION_CYCLES,
    MAX_MISSION_FINDINGS,
    MAX_OUTBOUND_MESSAGE_BYTES,
    MAX_POLYGON_POINTS,
    MAX_TARGETS,
    MAX_TARGET_CHARS,
    MAX_TOOL_CALLS_PER_CYCLE,
    MAX_TOOL_CALLS_PER_WINDOW,
    MAX_TOOL_ITEMS,
    MAX_TOOL_QUESTION_CHARS,
    MISSION_SCHEMA_VERSION,
    MISSION_STATES,
    MODE_INSPECT,
    MODE_WATCH,
    PERCEPTION_TOOL_NAMES,
    PROFILE_ID,
    PROFILE_VERSION,
    SCOPE_MISSION_CONTROL,
    SCOPE_MISSION_EVIDENCE,
    SCOPE_MISSION_READ,
    SCOPE_MISSION_REVIEW,
    Clock,
    Decision,
    EventSink,
    FindingProposal,
    GeometryItem,
    InputTransform,
    MissionEvent,
    MissionTool,
    Planner,
    PlannerContext,
    Principal,
    QualifiedModels,
    SourceBinding,
    SourceFrame,
    SourceProvider,
    ToolCallRecord,
    ToolContext,
    ToolRequest,
    ToolResult,
    USAGE_WINDOW_MS,
    WatchAdapter,
    WatchLease,
    WatchLeaseRelease,
    WatchLeaseRequest,
    WatchProposal,
    WATCH_LEASE_SECONDS,
    WATCH_MAX_AGE_SECONDS,
    WATCH_MIN_INTERVAL_SECONDS,
)
from .mission_store import (
    EvidenceUnavailable,
    IdempotencyConflict,
    InvalidEvidence,
    MissionStore,
    QuotaAccountingIncomplete,
    QuotaExceeded,
    RootQuotaExceeded,
    RevisionConflict,
    _jpeg_dimensions,
)
from .mission_record_projection import MissionRecordProjectionError, project_mission_records
from .mission_records import Record, RecordValidationError, record_to_dict
from .mission_packet_preview import PacketPreview, build_packet_preview

DEFAULT_GOAL = "Identify visible items relevant to this expertise and explain what needs a closer look."
MODEL_KEYS = ("gemma", "lfm", "lfm3b")
MODEL_LABELS = {
    "gemma": "Gemma 4 E2B",
    "lfm": "LFM2.5-VL 450M",
    "lfm3b": "LFM2.5-VL 3B",
}
ACTIVE_STATES = frozenset({"running", "waiting_evidence", "waiting_approval"})
MAX_MISSION_LIST_ITEMS = 20
SHUTDOWN_DRAIN_SECONDS = 2.0


class _SystemClock:
    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def monotonic(self) -> float:
        return time.monotonic()


class _CommandError(Exception):
    def __init__(self, code: str, message: str, *, result: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.result = dict(result or {})


class _NativeDeadlineExpired(Exception):
    """Native call drained after its mission budget and must not be committed."""


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(child) for child in value]
    return value


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _clean_text(value: Any, *, limit: int, field: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise _CommandError("invalid_request", f"{field} must be text")
    text = value.strip()
    if (required and not text) or len(text) > limit:
        raise _CommandError("invalid_request", f"{field} is empty or too long")
    return text


def _watch_task(args: Mapping[str, Any]) -> str | None:
    if "watch_task" not in args:
        return None
    value = args["watch_task"]
    if value not in ("detect", "segment"):
        raise _CommandError("invalid_request", "watch_task must be detect or segment")
    return value


def _source_binding(value: Any) -> SourceBinding:
    if isinstance(value, SourceBinding):
        source_id, source_epoch = value.source_id, value.source_epoch
    elif isinstance(value, Mapping) and set(value) == {"source_id", "source_epoch"}:
        source_id, source_epoch = value.get("source_id"), value.get("source_epoch")
    else:
        raise _CommandError("invalid_request", "source_binding must include source_id and source_epoch")
    if not isinstance(source_id, str) or not source_id or len(source_id) > 128:
        raise _CommandError("invalid_request", "source_binding source_id is invalid")
    if not isinstance(source_epoch, str) or not source_epoch or len(source_epoch) > 128:
        raise _CommandError("invalid_request", "source_binding source_epoch is invalid")
    return SourceBinding(source_id, source_epoch)


def _brief_sha256(expertise: str, goal: str) -> str:
    canonical = json.dumps(
        {"expertise": expertise, "goal": goal},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _brief_history_entry(version: int, expertise: str, goal: str, now_ms: int) -> dict[str, Any]:
    entry = {
        "version": int(version),
        "updated_at_ms": int(now_ms),
        "brief_sha256": _brief_sha256(expertise, goal),
        "expertise": expertise,
        "goal": goal,
    }
    return entry


def _transform(value: Any) -> InputTransform | None:
    if value is None:
        return None
    allowed = {"origin_width", "origin_height", "rotation_degrees", "resized", "scale_x", "scale_y", "flipped", "width", "height"}
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise _CommandError("invalid_request", "input_transform fields are invalid")
    width, height = value.get("origin_width"), value.get("origin_height")
    normalized_width, normalized_height = value.get("width"), value.get("height")
    rotation = value.get("rotation_degrees", 0)
    resized = value.get("resized", False)
    flipped = value.get("flipped", False)
    scale_x, scale_y = value.get("scale_x"), value.get("scale_y")
    if width is not None and (not _integer(width) or not 1 <= width <= 100_000):
        raise _CommandError("invalid_request", "input_transform origin_width is invalid")
    if height is not None and (not _integer(height) or not 1 <= height <= 100_000):
        raise _CommandError("invalid_request", "input_transform origin_height is invalid")
    if normalized_width is not None and (not _integer(normalized_width) or not 1 <= normalized_width <= 100_000):
        raise _CommandError("invalid_request", "input_transform width is invalid")
    if normalized_height is not None and (not _integer(normalized_height) or not 1 <= normalized_height <= 100_000):
        raise _CommandError("invalid_request", "input_transform height is invalid")
    if not _integer(rotation) or rotation % 90 or not 0 <= rotation < 360:
        raise _CommandError("invalid_request", "input_transform rotation must be 0, 90, 180, or 270")
    if not isinstance(resized, bool):
        raise _CommandError("invalid_request", "input_transform resized must be boolean")
    if not isinstance(flipped, bool):
        raise _CommandError("invalid_request", "input_transform flipped must be boolean")
    for scale in (scale_x, scale_y):
        if scale is not None and (isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1):
            raise _CommandError("invalid_request", "input_transform scale must be in (0, 1]")
    return InputTransform(width, height, rotation, resized, scale_x, scale_y, flipped, normalized_width, normalized_height)


def _geometry(value: Any) -> GeometryItem:
    if isinstance(value, GeometryItem):
        item = value
    elif isinstance(value, Mapping):
        item = GeometryItem(
            item_id=value.get("item_id", ""),
            label=value.get("label", ""),
            score=value.get("score"),
            box=tuple(value.get("box", ())),
            polygon=tuple(tuple(point) for point in value["polygon"]) if value.get("polygon") is not None else None,
            source=value.get("source", "visionbrain"),
        )
    else:
        raise ValueError("tool geometry item has invalid shape")
    if not isinstance(item.item_id, str) or not item.item_id or len(item.item_id) > 128:
        raise ValueError("tool geometry item id is invalid")
    if not isinstance(item.label, str) or not item.label.strip() or len(item.label) > 128:
        raise ValueError("tool geometry label is invalid")
    if not isinstance(item.source, str) or not item.source or len(item.source) > 128:
        raise ValueError("tool geometry source is invalid")
    if isinstance(item.score, bool) or not isinstance(item.score, (int, float)) or not math.isfinite(item.score) or not 0 <= item.score <= 1:
        raise ValueError("tool geometry score is invalid")
    if len(item.box) != 4 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in item.box):
        raise ValueError("tool geometry box is invalid")
    if item.box[2] <= item.box[0] or item.box[3] <= item.box[1]:
        raise ValueError("tool geometry box has no area")
    if item.polygon is not None:
        if not 3 <= len(item.polygon) <= MAX_POLYGON_POINTS:
            raise ValueError("tool geometry polygon is outside the point limit")
        if any(len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in point) for point in item.polygon):
            raise ValueError("tool geometry polygon contains invalid coordinates")
    return GeometryItem(item.item_id, item.label.strip(), float(item.score), tuple(float(x) for x in item.box), item.polygon, item.source)


def _geometry_dict(item: GeometryItem) -> dict[str, Any]:
    return {
        "item_id": item.item_id,
        "label": item.label,
        "score": item.score,
        "box": list(item.box),
        "polygon": [list(point) for point in item.polygon] if item.polygon else None,
        "source": item.source,
    }


def _record_from_dict(value: Mapping[str, Any]) -> ToolCallRecord:
    return ToolCallRecord(
        tool_result_id=str(value["tool_result_id"]),
        tool=str(value["tool"]),
        status=value["status"],
        input_evidence_id=str(value["input_evidence_id"]),
        items=tuple(_geometry(item) for item in value.get("items", ())),
        evidence_ids=tuple(str(item) for item in value.get("evidence_ids", ())),
        text=str(value.get("text", "")),
        error_code=value.get("error_code"),
        brief_version=int(value.get("brief_version", 1)),
        brief_sha256=value.get("brief_sha256"),
        model_provenance=dict(value.get("model_provenance", {})),
        unavailable_evidence_ids=tuple(
            str(item) for item in value.get("unavailable_evidence_ids", ())
        ),
        source_binding=(
            dict(value["source_binding"])
            if isinstance(value.get("source_binding"), Mapping)
            else None
        ),
        frame_id=value.get("frame_id"),
        input_sha256=value.get("input_sha256"),
        evidence_sha256=dict(value.get("evidence_sha256", {})),
        created_at_ms=value.get("created_at_ms"),
    )


def _validate_action(decision: Decision, *, allowed_tools: set[str], grounded_items: Mapping[str, GeometryItem]) -> tuple[str, dict[str, Any]]:
    if not isinstance(decision, Decision) or decision.schema_version != MISSION_SCHEMA_VERSION:
        raise ValueError("planner did not return a mission.v1 decision")
    tool, args = decision.tool, dict(decision.arguments)
    if tool not in allowed_tools:
        raise ValueError("planner selected a tool outside the advertised set")
    if tool in {"detect_objects", "segment_objects"}:
        if set(args) != {"targets"} or not isinstance(args["targets"], (list, tuple)):
            raise ValueError("target tool arguments are malformed")
        limit = 4 if tool == "segment_objects" else MAX_TARGETS
        if not 1 <= len(args["targets"]) <= limit:
            raise ValueError("target count is outside the tool limit")
        targets, seen = [], set()
        for raw in args["targets"]:
            target = _clean_text(raw, limit=MAX_TARGET_CHARS, field="target")
            if target.casefold() not in seen:
                seen.add(target.casefold())
                targets.append(target)
        args = {"targets": targets}
    elif tool in {"inspect_crop", "read_text"}:
        expected = {"item_id", "question"} if tool == "inspect_crop" else {"item_id", "question"}
        if set(args) - expected or "item_id" not in args:
            raise ValueError("grounded crop arguments are malformed")
        item_id = _clean_text(args["item_id"], limit=128, field="item_id")
        if item_id not in grounded_items:
            raise ValueError("crop item is not grounded in this cycle")
        question = args.get("question", "Read only clearly legible text verbatim." if tool == "read_text" else None)
        if question is None and tool == "inspect_crop":
            raise ValueError("inspect_crop requires a question")
        question = _clean_text(question, limit=MAX_TOOL_QUESTION_CHARS, field="question")
        args = {"item_id": item_id, "question": question}
    elif tool == "request_closeup":
        if set(args) not in ({"item_id", "reason"}, {"description", "reason"}):
            raise ValueError("close-up requires a grounded item or description and reason")
        if "item_id" in args:
            item_id = _clean_text(args["item_id"], limit=128, field="item_id")
            if item_id not in grounded_items:
                raise ValueError("close-up item is not grounded in this cycle")
            args["item_id"] = item_id
        else:
            args["description"] = _clean_text(args["description"], limit=500, field="description")
        args["reason"] = _clean_text(args["reason"], limit=500, field="reason")
    elif tool == "finish":
        if args:
            raise ValueError("finish does not accept arguments")
    else:
        raise ValueError("unknown mission action")
    return tool, args


class MissionRuntime:
    """One durable visual-inspection authority with bounded Inspect and Watch."""

    def __init__(
        self,
        store: MissionStore,
        planner: Planner,
        tools: MissionTool,
        source_provider: SourceProvider,
        watch_adapter: WatchAdapter,
        event_sink: EventSink,
        qualified_models: QualifiedModels | None = None,
        evaluation_overrides: QualifiedModels | None = None,
        approved_source_ids: Collection[str] = (),
        clock: Clock | None = None,
    ) -> None:
        self.store = store
        self.planner = planner
        self.tools = tools
        self.source_provider = source_provider
        self.watch_adapter = watch_adapter
        self.event_sink = event_sink
        self.qualified_models = {key: frozenset(modes) for key, modes in (qualified_models or {}).items()}
        self.evaluation_overrides = {key: frozenset(modes) for key, modes in (evaluation_overrides or {}).items()}
        self.approved_source_ids = frozenset(str(value) for value in approved_source_ids)
        self.clock: Clock = clock or _SystemClock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._task_interrupts: dict[str, asyncio.Event] = {}
        self._pending_generations: dict[str, int] = {}
        self._native_calls: set[asyncio.Task[Any]] = set()
        self._closeup_timers: dict[str, asyncio.Task[None]] = {}
        self._lease_timers: dict[str, asyncio.Task[None]] = {}
        self._close_cleanup: asyncio.Task[None] | None = None
        self._planner_closed = False
        self._planner_close_lock = asyncio.Lock()
        self._recovery_lock = asyncio.Lock()
        self._recovered = False
        self._closed = False

    async def get_record_bundle(self, mission_id: str) -> tuple[Record, ...]:
        """Verify referenced media, then project one persisted mission record bundle."""
        await self._ensure_recovered()

        def verify_snapshot(tx):
            snapshot = tx.get_mission(mission_id)
            if snapshot is None:
                raise KeyError(mission_id)
            self._verify_snapshot_evidence(mission_id, snapshot, tx=tx, verify_content=True)
            rows = self.store.read_mission_record_rows(mission_id, tx=tx)
            if rows is None:
                raise KeyError(mission_id)
            return rows

        verification = await asyncio.to_thread(self.store.perform_read, verify_snapshot)
        self._publish_events(verification.events)
        rows = verification.reply
        return await asyncio.to_thread(
            project_mission_records,
            rows["snapshot"],
            rows["evidence_rows"],
            rows["tool_rows"],
        )

    async def preview_packet(self, mission_id: str) -> PacketPreview:
        """Return a draft packet from persisted metadata without fresh media verification."""
        rows = await asyncio.to_thread(self.store.read_mission_record_rows, mission_id)
        if rows is None:
            raise KeyError(mission_id)
        return await asyncio.to_thread(build_packet_preview, rows)

    def _model_status(self, key: str) -> tuple[bool, str | None, str | None, str | None]:
        """Return local readiness and checkpoint provenance without loading models."""
        checkpoint = checkpoint_revision = failure = None
        try:
            from .mission_models import _cached_checkpoint_path, _model_target

            checkpoint = _model_target(key)
            cached_path = _cached_checkpoint_path(checkpoint)
            checkpoint_revision = cached_path.name if cached_path is not None else None
        except Exception as exc:
            failure = getattr(exc, "error_code", None) or "model_unavailable"
        try:
            ready = bool(self.planner.available(key))
        except Exception:
            ready = False
        return ready, checkpoint, checkpoint_revision, failure

    def _model_provenance(self, key: str, mode: str) -> dict[str, Any]:
        ready, checkpoint, checkpoint_revision, failure = self._model_status(key)
        qualified = mode in self.qualified_models.get(key, frozenset())
        evaluation = mode in self.evaluation_overrides.get(key, frozenset()) and not qualified
        provenance = {
            "model_key": key,
            "checkpoint": checkpoint,
            "qualification": "qualified" if qualified else "unqualified",
            "authorization_basis": (
                "operator_evaluation_override" if evaluation
                else "measured_qualification" if qualified
                else None
            ),
        }
        if checkpoint_revision:
            provenance["checkpoint_revision"] = checkpoint_revision
        if failure:
            provenance["availability_error"] = failure
        if not ready:
            provenance["ready"] = False
        return provenance

    def capabilities(self, *, integrated: bool = False) -> dict[str, Any]:
        """Return qualified modes and, only for negotiated clients, evaluation routes."""
        available_tools = set()
        try:
            available_tools = set(self.tools.available_tools())
        except Exception:
            pass
        models = []
        supported_modes: set[str] = set()
        for key in MODEL_KEYS:
            ready, checkpoint, checkpoint_revision, failure = self._model_status(key)
            modes = sorted(self.qualified_models.get(key, frozenset()) & {MODE_INSPECT, MODE_WATCH}) if ready else []
            evaluation_modes = sorted(
                (self.evaluation_overrides.get(key, frozenset()) - self.qualified_models.get(key, frozenset()))
                & {MODE_INSPECT, MODE_WATCH}
            ) if ready and integrated else []
            supported_modes.update(modes)
            supported_modes.update(evaluation_modes)
            model = {"key": key, "label": MODEL_LABELS[key], "modes": modes, "ready": ready}
            if failure:
                model["availability_error"] = failure
            if integrated and evaluation_modes:
                model.update({
                    "evaluation_modes": evaluation_modes,
                    "qualification": "unqualified",
                    "authorization_basis": "operator_evaluation_override",
                    "model_key": key,
                })
                if checkpoint:
                    model["checkpoint"] = checkpoint
                if checkpoint_revision:
                    model["checkpoint_revision"] = checkpoint_revision
            models.append(model)
        tool_names: set[str] = set()
        if supported_modes:
            tool_names.update(available_tools & PERCEPTION_TOOL_NAMES)
            tool_names.update({"finish", "request_closeup"})
        from .mission_models import MISSION_TOOL_SCHEMAS

        tools = [dict(schema) for schema in MISSION_TOOL_SCHEMAS if schema["name"] in tool_names]
        sources = []
        owner = getattr(self.source_provider, "__self__", None)
        listing = getattr(self.source_provider, "list_sources", None) or getattr(owner, "list_sources", None)
        if callable(listing):
            try:
                sources = [_jsonable(item) for item in listing()]
            except Exception:
                sources = []
        if integrated:
            sources = [
                source for source in sources
                if isinstance(source, Mapping) and source.get("source_id") in self.approved_source_ids
            ]
        available = bool(supported_modes)
        descriptor = {
            "available": available,
            "detail": None if available else "no installed model/mode pair is authorized",
            "schema_version": MISSION_SCHEMA_VERSION,
            "profiles": [PROFILE_ID],
            "profile_versions": {PROFILE_ID: PROFILE_VERSION},
            "modes": sorted(supported_modes),
            "models": models,
            "tools": [schema["name"] for schema in tools],
            "tool_schemas": tools,
            "limits": {
                "decoded_jpeg_bytes": MAX_DECODED_JPEG_BYTES,
                "outbound_message_bytes": MAX_OUTBOUND_MESSAGE_BYTES,
                "generations_per_cycle": MAX_GENERATIONS_PER_CYCLE,
                "tool_calls_per_cycle": MAX_TOOL_CALLS_PER_CYCLE,
                "active_seconds": MAX_ACTIVE_SECONDS,
                "human_wait_seconds": MAX_HUMAN_WAIT_SECONDS,
                "generations_per_five_minutes": MAX_GENERATIONS_PER_WINDOW,
                "tool_calls_per_five_minutes": MAX_TOOL_CALLS_PER_WINDOW,
                "mission_evidence_records": MAX_MISSION_EVIDENCE,
                "mission_evidence_bytes": int(getattr(self.store, "quota_bytes", 2 * 1024 * 1024 * 1024)),
                "mission_evidence_history_records": MAX_MISSION_EVIDENCE_HISTORY,
                "brief_versions": MAX_BRIEF_VERSIONS,
                "watch_cycles": MAX_MISSION_CYCLES,
            },
            "sources": sources,
        }
        if integrated:
            descriptor["integrated_commands"] = ["prepare", "activate", "update_brief"]
        return descriptor

    async def handle(
        self,
        command: Mapping[str, Any],
        principal: Principal | str,
        scopes: Collection[str],
        *,
        integrated: bool = False,
    ) -> dict[str, Any]:
        """Validate authorization and idempotently handle one mission command."""
        if not isinstance(command, Mapping) or command.get("command") != "preview_packet":
            await self._ensure_recovered()
        if self._closed:
            return self._reply(command if isinstance(command, Mapping) else {}, False, error=("runtime_closed", "mission runtime is closed"))
        if not isinstance(command, Mapping):
            return self._reply({}, False, error=("invalid_request", "mission command must be an object"))
        request_id = command.get("request_id")
        if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
            return self._reply(command, False, error=("invalid_request", "request_id is required and bounded"))
        try:
            normalized = self._validate_envelope(command)
            if normalized["command"] in {"prepare", "activate", "update_brief"} and not integrated:
                raise _CommandError("unsupported_capability", "command requires mission.integrated.v1")
            principal_id = principal.principal_id if isinstance(principal, Principal) else principal
            if not isinstance(principal_id, str) or not principal_id.strip() or len(principal_id) > 256:
                raise _CommandError("unauthorized", "authenticated principal is missing")
            required = self._required_scopes(normalized["command"])
            if not required.issubset(set(scopes)):
                raise _CommandError("unauthorized", "mission credential lacks the required scope")
            if normalized["command"] == "capabilities":
                return self._reply(normalized, True, {"descriptor": self.capabilities(integrated=integrated)})
            payload = dict(normalized)
            followup: list[tuple[str, int]] = []
            release_after: list[tuple[Mapping[str, Any], str]] = []
            applied_watch_leases: list[WatchLease] = []

            async def cleanup_applied_watch_leases() -> None:
                leases = tuple(applied_watch_leases)
                applied_watch_leases.clear()
                for lease in leases:
                    await asyncio.shield(
                        self._release_lease(
                            asdict(lease),
                            "watch_task_update_not_committed",
                            cancel_timer=False,
                        )
                    )

            preflight: dict[str, Any] = {
                "draining_missions": {
                    key for key, task in self._tasks.items()
                    if key != normalized.get("mission_id") and not task.done()
                }
            }
            if normalized["command"] == "create":
                preflight["configuration_revision"] = self._configuration_revision()
            elif normalized["command"] in {"resume", "activate", "update_brief"}:
                mission_id = normalized["mission_id"]
                prior_task = self._tasks.get(mission_id)
                preflight["same_mission_busy"] = prior_task is not None and not prior_task.done()
                current = await asyncio.to_thread(self.store.get_mission, mission_id)
                if current is not None:
                    preflight["planner_ready"] = self._model_status(
                        current.get("reasoning_model")
                    )[0]
                    preflight["current_snapshot"] = current
                source_value = normalized["args"].get("source_binding")
                if normalized["command"] == "update_brief" and current is not None:
                    source_value = current.get("source_binding")
                    # Prepared missions hold only an approved source_id until
                    # the first explicit activation. They must remain waiting
                    # without attempting to resolve a live source epoch.
                    if not (
                        isinstance(source_value, Mapping)
                        and set(source_value) == {"source_id", "source_epoch"}
                    ):
                        source_value = None
                if source_value is not None:
                    binding = _source_binding(source_value)
                    preflight["source_frame"] = self.source_provider(binding)
                    preflight["configuration_revision"] = self._configuration_revision()

            def operation(tx):
                try:
                    result = self._command_operation(
                        tx,
                        normalized,
                        principal if isinstance(principal, Principal) else Principal(principal_id),
                        followup,
                        release_after,
                        preflight,
                        applied_watch_leases,
                        self.clock.now_ms(),
                    )
                    return self._reply(normalized, True, result)
                except _CommandError as exc:
                    return self._reply(normalized, False, exc.result, (exc.code, str(exc)))

            if normalized["command"] in {"get", "list", "events_since", "export", "get_evidence", "preview_packet"}:
                worker = asyncio.create_task(asyncio.to_thread(self.store.perform_read, operation))
            else:
                worker = asyncio.create_task(asyncio.to_thread(
                    self.store.perform_request,
                    principal_id,
                    normalized["request_id"],
                    payload,
                    operation,
                    now_ms=self.clock.now_ms(),
                ))
            cancelled = False
            try:
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        if worker.cancelled():
                            raise
                        cancelled = True
                saved = worker.result()
            except BaseException:
                await cleanup_applied_watch_leases()
                raise
            if saved.reply.get("ok"):
                # The durable reply now owns the applied lease; the existing
                # success path below installs its returned expiry timer.
                applied_watch_leases.clear()
            else:
                await cleanup_applied_watch_leases()
            self._publish_events(saved.events)
            if not saved.replayed:
                if saved.reply.get("ok"):
                    current_snapshot = saved.reply.get("result", {}).get("snapshot")
                    if isinstance(current_snapshot, Mapping):
                        if current_snapshot.get("closeup_request") is None:
                            self._cancel_timer(self._closeup_timers, current_snapshot["mission_id"])
                        if current_snapshot.get("watch_lease") is None:
                            self._cancel_timer(self._lease_timers, current_snapshot["mission_id"])
                        elif (
                            normalized["command"] == "update_brief"
                            and "watch_task" in normalized["args"]
                            and current_snapshot.get("state") == "running"
                        ):
                            lease_data = current_snapshot["watch_lease"]
                            lease = WatchLease(
                                lease_id=str(lease_data["lease_id"]),
                                mission_id=str(lease_data["mission_id"]),
                                source_binding=_source_binding(lease_data["source_binding"]),
                                configuration_revision=int(lease_data["configuration_revision"]),
                                targets=tuple(lease_data["targets"]),
                                task=str(lease_data["task"]),
                                expires_at_ms=int(lease_data["expires_at_ms"]),
                            )
                            self._schedule_lease_timeout(
                                str(current_snapshot["mission_id"]),
                                int(current_snapshot["execution_generation"]),
                                lease,
                            )
                    if normalized["command"] in {"pause", "cancel", "decline_closeup"}:
                        self._interrupt_watch_pacing(normalized["mission_id"])
                for mission_id, generation in followup:
                    self._interrupt_watch_pacing(mission_id)
                for lease, reason in release_after:
                    await self._release_lease(lease, reason)
                for mission_id, generation in followup:
                    self._start_task(mission_id, generation)
            if cancelled:
                raise asyncio.CancelledError
            return saved.reply
        except IdempotencyConflict as exc:
            return self._reply(command, False, {}, ("invalid_request", str(exc)))
        except RevisionConflict as exc:
            return self._reply(command, False, {"snapshot": exc.snapshot}, ("revision_conflict", "mission revision changed"))
        except (InvalidEvidence, EvidenceUnavailable) as exc:
            if normalized.get("command") == "get_evidence":
                evidence_id = normalized["args"].get("evidence_id")
                owner = await asyncio.to_thread(self.store.evidence_owner, evidence_id)
                if owner == normalized.get("mission_id"):
                    await asyncio.to_thread(
                        self._mark_evidence_unavailable,
                        owner,
                        evidence_id,
                        getattr(exc, "availability_reason", "corrupt"),
                    )
            return self._reply(command, False, {}, ("evidence_unavailable", str(exc)))
        except QuotaAccountingIncomplete:
            return self._reply(
                command,
                False,
                {},
                (
                    "evidence_quota_accounting_unavailable",
                    "Evidence storage could not be accounted safely; resolve the storage issue and retry.",
                ),
            )
        except QuotaExceeded as exc:
            return self._reply(command, False, {}, ("quota_exceeded", str(exc)))
        except KeyError:
            return self._reply(command, False, {}, ("mission_not_found", "mission or evidence was not found"))
        except _CommandError as exc:
            return self._reply(command, False, exc.result, (exc.code, str(exc)))
        except (TypeError, ValueError, binascii.Error) as exc:
            return self._reply(command, False, {}, ("invalid_request", str(exc)))

    def _validate_envelope(self, command: Mapping[str, Any]) -> dict[str, Any]:
        if set(command) - {"type", "schema_version", "request_id", "command", "mission_id", "expected_revision", "args"}:
            raise _CommandError("invalid_request", "mission command contains unknown envelope fields")
        if command.get("type") != "mission_command" or command.get("schema_version") != MISSION_SCHEMA_VERSION:
            raise _CommandError("invalid_request", "mission.v1 command envelope is invalid")
        name = command.get("command")
        if name not in {"capabilities", "create", "prepare", "activate", "update_brief", "attach_evidence", "resume", "pause", "cancel", "get", "list", "events_since", "get_evidence", "review_finding", "decline_closeup", "export", "preview_packet"}:
            raise _CommandError("invalid_request", "unknown mission command")
        mission_id = command.get("mission_id")
        if name in {"create", "prepare", "list", "capabilities"}:
            if mission_id is not None:
                raise _CommandError("invalid_request", f"{name} does not accept mission_id")
        elif not isinstance(mission_id, str) or not mission_id.strip() or len(mission_id) > 128:
            raise _CommandError("invalid_request", "mission_id is required")
        if name == "prepare" and "expected_revision" in command:
            raise _CommandError("invalid_request", "prepare does not accept expected_revision")
        args = command.get("args", {})
        if not isinstance(args, Mapping):
            raise _CommandError("invalid_request", "args must be an object")
        if name == "preview_packet" and ("args" not in command or bool(args)):
            raise _CommandError("invalid_request", "preview_packet args must be an empty object")
        normalized = dict(command)
        normalized["args"] = dict(args)
        if "expected_revision" in command and not _integer(command["expected_revision"]):
            raise _CommandError("invalid_request", "expected_revision must be an integer")
        if name == "preview_packet" and (
            not _integer(command.get("expected_revision")) or command["expected_revision"] < 0
        ):
            raise _CommandError("invalid_request", "preview_packet requires a non-negative expected_revision")
        if name in {"attach_evidence", "activate", "update_brief", "resume", "pause", "cancel", "review_finding", "decline_closeup"}:
            if not _integer(command.get("expected_revision")):
                raise _CommandError("invalid_request", f"{name} requires expected_revision")
        return normalized

    @staticmethod
    def _fresh_budget(now_ms: int) -> dict[str, int]:
        return {
            "generations_remaining": MAX_GENERATIONS_PER_CYCLE,
            "tools_remaining": MAX_TOOL_CALLS_PER_CYCLE,
            "window_started_at_ms": int(now_ms),
            "window_generations_used": 0,
            "window_tools_used": 0,
            "window_generations_remaining": MAX_GENERATIONS_PER_WINDOW,
            "window_tools_remaining": MAX_TOOL_CALLS_PER_WINDOW,
        }

    @classmethod
    def _budget_for_resume(cls, value: Any, now_ms: int) -> dict[str, int]:
        prior = dict(value) if isinstance(value, Mapping) else {}
        started = prior.get("window_started_at_ms")
        if not _integer(started) or now_ms - started >= USAGE_WINDOW_MS or now_ms < started:
            return cls._fresh_budget(now_ms)
        generations = prior.get("window_generations_used", 0)
        tools = prior.get("window_tools_used", 0)
        generations = generations if _integer(generations) else 0
        tools = tools if _integer(tools) else 0
        return {
            "generations_remaining": MAX_GENERATIONS_PER_CYCLE,
            "tools_remaining": MAX_TOOL_CALLS_PER_CYCLE,
            "window_started_at_ms": started,
            "window_generations_used": generations,
            "window_tools_used": tools,
            "window_generations_remaining": max(0, MAX_GENERATIONS_PER_WINDOW - generations),
            "window_tools_remaining": max(0, MAX_TOOL_CALLS_PER_WINDOW - tools),
        }

    @classmethod
    def _budget_for_dispatch(cls, value: Any, now_ms: int) -> dict[str, int]:
        prior = dict(value) if isinstance(value, Mapping) else {}
        cycle_generations = prior.get("generations_remaining", MAX_GENERATIONS_PER_CYCLE)
        cycle_tools = prior.get("tools_remaining", MAX_TOOL_CALLS_PER_CYCLE)
        if not _integer(cycle_generations):
            cycle_generations = MAX_GENERATIONS_PER_CYCLE
        if not _integer(cycle_tools):
            cycle_tools = MAX_TOOL_CALLS_PER_CYCLE
        started = prior.get("window_started_at_ms")
        if not _integer(started) or now_ms - started >= USAGE_WINDOW_MS or now_ms < started:
            prior.update(cls._fresh_budget(now_ms))
        else:
            prior.setdefault("window_generations_used", 0)
            prior.setdefault("window_tools_used", 0)
            prior.setdefault("window_generations_remaining", MAX_GENERATIONS_PER_WINDOW)
            prior.setdefault("window_tools_remaining", MAX_TOOL_CALLS_PER_WINDOW)
        prior["generations_remaining"] = cycle_generations
        prior["tools_remaining"] = cycle_tools
        return prior

    @staticmethod
    def _required_scopes(command: str) -> frozenset[str]:
        if command in {"capabilities", "get", "list", "events_since", "export", "preview_packet"}:
            return frozenset({SCOPE_MISSION_READ})
        if command == "get_evidence":
            return frozenset({SCOPE_MISSION_READ, SCOPE_MISSION_EVIDENCE})
        if command == "review_finding":
            return frozenset({SCOPE_MISSION_REVIEW})
        if command == "attach_evidence":
            return frozenset({SCOPE_MISSION_CONTROL, SCOPE_MISSION_EVIDENCE})
        return frozenset({SCOPE_MISSION_CONTROL})

    def _command_operation(
        self,
        tx,
        command: Mapping[str, Any],
        principal: Principal,
        followup: list[tuple[str, int]],
        release_after: list[tuple[Mapping[str, Any], str]],
        preflight: Mapping[str, Any],
        applied_watch_leases: list[WatchLease],
        now: int,
    ) -> dict[str, Any]:
        name, args = command["command"], command["args"]
        mission_id = command.get("mission_id")
        if name == "list":
            limit = args.get("limit", 50)
            before = args.get("before_updated_at_ms")
            if not _integer(limit) or (before is not None and not _integer(before)):
                raise _CommandError("invalid_request", "list cursor or limit is invalid")
            missions = self.store.list_missions(limit=min(limit, MAX_MISSION_LIST_ITEMS), before_updated_at_ms=before)
            return {"missions": [self._mission_summary(item) for item in missions]}
        if name == "prepare":
            allowed = {"profile", "expertise", "goal", "mode", "reasoning_model", "source_id", "watch_task"}
            if set(args) - allowed:
                raise _CommandError("invalid_request", "prepare contains unknown fields")
            watch_task = _watch_task(args)
            if args.get("profile") != {"id": PROFILE_ID, "version": PROFILE_VERSION}:
                raise _CommandError("unsupported_profile", "visual_inspection version 1 is the only profile")
            source_id = args.get("source_id")
            if (
                not isinstance(source_id, str)
                or not source_id
                or len(source_id) > 128
                or source_id not in self.approved_source_ids
            ):
                raise _CommandError("source_not_approved", "source_id is not in the server approved-source registry")
            expertise = _clean_text(args.get("expertise"), limit=MAX_EXPERTISE_CHARS, field="expertise")
            goal = _clean_text(args.get("goal", DEFAULT_GOAL), limit=MAX_GOAL_CHARS, field="goal")
            mode, model = args.get("mode", MODE_WATCH), args.get("reasoning_model", "gemma")
            if mode not in {MODE_INSPECT, MODE_WATCH} or model not in MODEL_KEYS:
                raise _CommandError("invalid_request", "mode or reasoning_model is unsupported")
            mission_id = uuid.uuid4().hex
            provenance = self._model_provenance(model, mode)
            snapshot = {
                "mission_id": mission_id,
                "revision": 1,
                "state": "created",
                "reason": None,
                "expertise": expertise,
                "goal": goal,
                "brief_version": 1,
                "brief_history": [_brief_history_entry(1, expertise, goal, now)],
                "brief_sha256": _brief_sha256(expertise, goal),
                "mode": mode,
                "reasoning_model": model,
                "model_provenance": provenance,
                "source_binding": {"source_id": source_id},
                "activation_intent": "when_source_starts",
                "input_evidence_id": None,
                "activity": "Mission prepared. Activate after the approved source starts streaming.",
                "targets": [],
                "task": "detect",
                "findings": [],
                "evidence": [],
                "cycle_history": [],
                "closeup_request": None,
                "budget": self._fresh_budget(now),
                "cycle_id": None,
                "execution_generation": 0,
                "configuration_revision": int(preflight.get("configuration_revision", 0)),
                "watch_lease": None,
                "last_sequence": 0,
                "updated_at_ms": now,
            }
            if watch_task is not None and mode == MODE_WATCH:
                snapshot["watch_task"] = watch_task
            tx.insert_mission(snapshot)
            tx.append_event(mission_id, 1, "mission_prepared", {"snapshot": snapshot}, now)
            return {"snapshot": tx.get_mission(mission_id), "brief_version": 1, "execution_outcome": "waiting_for_source"}
        if name == "create":
            allowed = {"profile", "expertise", "goal", "mode", "reasoning_model", "source_binding"}
            if set(args) - allowed:
                raise _CommandError("invalid_request", "create contains unknown fields")
            profile = args.get("profile")
            if profile != {"id": PROFILE_ID, "version": PROFILE_VERSION}:
                raise _CommandError("unsupported_profile", "visual_inspection version 1 is the only profile")
            expertise = _clean_text(args.get("expertise"), limit=MAX_EXPERTISE_CHARS, field="expertise")
            goal = _clean_text(args.get("goal", DEFAULT_GOAL), limit=MAX_GOAL_CHARS, field="goal")
            mode, model = args.get("mode", MODE_INSPECT), args.get("reasoning_model", "gemma")
            if mode not in {MODE_INSPECT, MODE_WATCH}:
                raise _CommandError("invalid_request", "mode must be inspect or watch")
            if model not in MODEL_KEYS:
                raise _CommandError("invalid_request", "reasoning_model is not a supported local model key")
            binding = _source_binding(args.get("source_binding")) if mode == MODE_WATCH else None
            configuration_revision = int(preflight.get("configuration_revision", 0))
            mission_id = uuid.uuid4().hex
            snapshot = {
                "mission_id": mission_id,
                "revision": 1,
                "state": "created",
                "reason": None,
                "expertise": expertise,
                "goal": goal,
                "brief_version": 1,
                "brief_history": [_brief_history_entry(1, expertise, goal, now)],
                "brief_sha256": _brief_sha256(expertise, goal),
                "mode": mode,
                "reasoning_model": model,
                "model_provenance": self._model_provenance(model, mode),
                "source_binding": asdict(binding) if binding else None,
                "input_evidence_id": None,
                "activity": "Mission created. Review readiness, then resume.",
                "targets": [],
                "task": "detect",
                "findings": [],
                "closeup_request": None,
                "evidence": [],
                "cycle_history": [],
                "budget": self._fresh_budget(now),
                "cycle_id": None,
                "execution_generation": 0,
                "configuration_revision": configuration_revision,
                "watch_lease": None,
                "last_sequence": 0,
                "updated_at_ms": now,
            }
            tx.insert_mission(snapshot)
            event = tx.append_event(mission_id, 1, "mission_created", {"snapshot": snapshot}, now)
            snapshot = tx.get_mission(mission_id)
            return {"snapshot": snapshot}

        if name == "get":
            snapshot = tx.get_mission(mission_id)
            if snapshot is None:
                raise KeyError(mission_id)
            return {"snapshot": self._verify_snapshot_evidence(mission_id, snapshot, tx=tx)}
        if name == "preview_packet":
            rows = self.store.read_mission_record_rows(mission_id, tx=tx)
            if rows is None:
                raise KeyError(mission_id)
            snapshot = rows["snapshot"]
            snapshot_revision = snapshot.get("revision")
            if not _integer(snapshot_revision) or snapshot_revision < 1:
                raise _CommandError(
                    "packet_projection_error",
                    "mission revision metadata is invalid",
                    result={
                        "mission_revision": None,
                        "projection_error": {"reason": "invalid_mission_revision", "path": "snapshot.revision"},
                    },
                )
            if snapshot_revision != command["expected_revision"]:
                raise RevisionConflict(snapshot)
            if snapshot.get("state") not in {"completed", "paused"}:
                raise _CommandError(
                    "packet_not_ready",
                    "mission must be completed or paused before packet preview",
                    result={
                        "mission_revision": snapshot_revision,
                        "mission_state": snapshot.get("state"),
                    },
                )
            if snapshot.get("watch_lease") is not None:
                raise _CommandError(
                    "packet_projection_error",
                    "mission still has an active Watch lease",
                    result={
                        "mission_revision": snapshot_revision,
                        "projection_error": {"reason": "active_watch_lease", "path": "snapshot.watch_lease"},
                    },
                )
            try:
                preview = build_packet_preview(rows)
            except MissionRecordProjectionError as exc:
                raise _CommandError(
                    "packet_projection_error",
                    "mission records cannot be projected into a packet",
                    result={
                        "mission_revision": snapshot_revision,
                        "projection_error": {"reason": exc.reason, "path": exc.path},
                    },
                ) from exc
            except RecordValidationError as exc:
                raise _CommandError(
                    "packet_projection_error",
                    "packet record bundle failed validation",
                    result={
                        "mission_revision": snapshot_revision,
                        "projection_error": {"reason": exc.reason, "path": exc.path},
                    },
                ) from exc
            except ValueError as exc:
                raise _CommandError(
                    "packet_projection_error",
                    "mission metadata cannot be projected into a packet",
                    result={
                        "mission_revision": snapshot_revision,
                        "projection_error": {"reason": "invalid_packet_projection", "path": ""},
                    },
                ) from exc
            return {
                "envelope_version": 1,
                "kind": "inspection_packet_preview",
                "verification": {
                    "basis": "persisted_metadata_only",
                    "fresh_media_verified": False,
                },
                "mission_id": preview.packet.mission_id,
                "mission_revision": preview.packet.mission_revision,
                "packet": record_to_dict(preview.packet),
                "bundle": [record_to_dict(record) for record in preview.records],
                "canonical_json": preview.canonical_json,
                "content_sha256": preview.content_sha256,
            }
        if name == "events_since":
            cursor = args.get("cursor", 0)
            limit = args.get("limit", 100)
            if not _integer(cursor) or not _integer(limit):
                raise _CommandError("invalid_request", "event cursor and limit must be integers")
            page = self.store.events_since(mission_id, cursor, limit=limit)
            # next_cursor intentionally remains the last returned event on a partial page.
            return page
        if name == "export":
            snapshot = tx.get_mission(mission_id)
            if snapshot is None:
                raise KeyError(mission_id)
            snapshot = self._verify_snapshot_evidence(mission_id, snapshot, tx=tx)
            packet = {
                "schema_version": MISSION_SCHEMA_VERSION,
                "mission": snapshot,
                "findings": snapshot.get("findings", []),
                "evidence": [self._public_evidence(item) for item in snapshot.get("evidence", [])],
            }
            referenced_tools = set()
            for finding in packet["findings"]:
                if not isinstance(finding, Mapping):
                    continue
                for item_ref in finding.get("item_refs", ()):
                    if isinstance(item_ref, Mapping) and isinstance(item_ref.get("tool_result_id"), str):
                        referenced_tools.add(item_ref["tool_result_id"])
                referenced_tools.update(
                    value for value in finding.get("text_refs", ())
                    if isinstance(value, str)
                )
            packet["tool_provenance"] = [
                {
                    key: record[key]
                    for key in (
                        "tool_result_id",
                        "tool",
                        "status",
                        "brief_version",
                        "brief_sha256",
                        "model_provenance",
                        "source_binding",
                        "frame_id",
                        "created_at_ms",
                        "input_evidence_id",
                        "input_sha256",
                        "evidence_ids",
                        "evidence_sha256",
                    )
                    if key in record
                }
                for record in self.store.tool_records(mission_id)
                if record.get("tool_result_id") in referenced_tools
            ]
            reply = self._reply(command, True, {"packet": packet})
            if reply.get("ok"):
                tx.pin_exported_evidence(
                    mission_id,
                    [
                        ref["evidence_id"]
                        for ref in packet["evidence"]
                        if ref.get("available", True) is True
                    ],
                )
            return {"packet": packet}
        if name == "get_evidence":
            evidence_id = args.get("evidence_id")
            offset, length = args.get("offset", 0), args.get("length", 49_152)
            if not isinstance(evidence_id, str) or not evidence_id or not _integer(offset) or not _integer(length):
                raise _CommandError("invalid_request", "evidence reference or chunk range is invalid")
            owner = self.store.evidence_owner(evidence_id)
            if owner != mission_id:
                raise _CommandError("invalid_request", "evidence does not belong to this mission")
            chunk = self.store.get_evidence_chunk(evidence_id, offset=offset, length=length)
            return {
                "evidence_id": chunk["evidence_id"],
                "sha256": chunk["sha256"],
                "total_bytes": chunk["total_bytes"],
                "offset": chunk["offset"],
                "width": chunk["width"],
                "height": chunk["height"],
                "jpeg_b64": base64.b64encode(chunk["jpeg_bytes"]).decode("ascii"),
            }
        if name == "attach_evidence":
            snapshot = self._require_revision(tx, mission_id, command)
            allowed = {"jpeg_b64", "sha256", "closeup_request_id", "capture_time_ms", "input_transform"}
            if set(args) - allowed or not isinstance(args.get("jpeg_b64"), str):
                raise _CommandError("invalid_request", "evidence attachment fields are invalid")
            encoded = args["jpeg_b64"]
            if len(encoded) > (MAX_DECODED_JPEG_BYTES * 4 // 3 + 8):
                raise _CommandError("invalid_request", "encoded JPEG exceeds the 2 MiB limit")
            try:
                jpeg = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise _CommandError("invalid_request", "jpeg_b64 is invalid base64") from exc
            if not jpeg or len(jpeg) > MAX_DECODED_JPEG_BYTES:
                raise _CommandError("invalid_request", "decoded JPEG must be non-empty and at most 2 MiB")
            sha = args.get("sha256")
            if not isinstance(sha, str) or len(sha) != 64:
                raise _CommandError("invalid_request", "sha256 must be a 64-character digest")
            transform = _transform(args.get("input_transform"))
            if principal.source_id is not None and (not isinstance(principal.source_id, str) or not principal.source_id or len(principal.source_id) > 128):
                raise _CommandError("invalid_request", "authenticated evidence source_id is invalid")
            if principal.source_epoch is not None and (not isinstance(principal.source_epoch, str) or not principal.source_epoch or len(principal.source_epoch) > 128):
                raise _CommandError("invalid_request", "authenticated evidence source_epoch is invalid")
            if principal.frame_id is not None and (not _integer(principal.frame_id) or not 0 <= principal.frame_id <= 2**63 - 1):
                raise _CommandError("invalid_request", "authenticated evidence frame_id is invalid")
            if principal.capture_time_ms is not None and (not _integer(principal.capture_time_ms) or not 0 <= principal.capture_time_ms <= 2**63 - 1):
                raise _CommandError("invalid_request", "authenticated evidence capture_time_ms is invalid")
            closeup_id = args.get("closeup_request_id")
            closeup = snapshot.get("closeup_request")
            is_closeup = snapshot.get("state") == "waiting_evidence" and closeup is not None
            if is_closeup:
                if closeup_id != closeup.get("request_id") or now > int(closeup.get("expires_at_ms", 0)):
                    raise _CommandError("closeup_mismatch", "photo does not match a current close-up request")
            elif closeup_id is not None:
                raise _CommandError("closeup_mismatch", "there is no current close-up request")
            if snapshot["state"] in {"completed", "cancelled", "failed"}:
                raise _CommandError("invalid_state", "terminal missions cannot accept evidence")
            binding = None
            if principal.source_id and principal.source_epoch:
                binding = SourceBinding(principal.source_id, principal.source_epoch)
            width, height = _jpeg_dimensions(jpeg)
            if transform and ((transform.width is not None and transform.width != width) or (transform.height is not None and transform.height != height)):
                raise _CommandError("invalid_request", "input_transform width and height must match the normalized JPEG")
            try:
                room_available = self._make_evidence_room(
                    tx, snapshot, len(jpeg), protected_evidence_ids=(snapshot.get("input_evidence_id"),)
                )
                if not room_available:
                    raise QuotaExceeded("no unprotected rolling evidence or byte headroom is available")
                evidence = tx.save_evidence(
                    mission_id,
                    jpeg,
                    kind=("closeup" if is_closeup else principal.evidence_kind if principal.evidence_kind in {"imported", "closeup", "frame"} else "imported"),
                    created_at_ms=now,
                    origin="explicit_attachment",
                    source_id=principal.source_id,
                    source_epoch=principal.source_epoch,
                    frame_id=principal.frame_id,
                    capture_time_ms=principal.capture_time_ms,
                    input_transform=asdict(transform) if transform else None,
                    closeup_request_id=closeup_id,
                    brief_version=int(snapshot.get("brief_version", 1)),
                    brief_sha256=snapshot.get("brief_sha256"),
                    model_provenance=snapshot.get("model_provenance", {}),
                    expected_sha256=sha,
                )
            except QuotaAccountingIncomplete:
                reason = "evidence_quota_accounting_unavailable"
                saved, lease = self._pause_for_evidence_quota(
                    tx, snapshot, now, reason=reason, preserve_closeup_wait=is_closeup
                )
                if lease:
                    release_after.append((lease, reason))
                self._replace_event_snapshot(tx, saved)
                raise _CommandError(
                    reason,
                    "Evidence storage could not be accounted safely; resolve the storage issue and retry the requested close-up."
                    if is_closeup else
                    "Evidence storage could not be accounted safely; resolve the storage issue and explicitly resume.",
                    result={"snapshot": saved, "execution_outcome": "waiting_evidence" if is_closeup else "paused"},
                )
            except RootQuotaExceeded as exc:
                self._roll_off_root_only_evidence(
                    tx,
                    snapshot,
                    exc.root_usage_bytes,
                    exc.incoming_bytes,
                    protected_evidence_ids=(snapshot.get("input_evidence_id"),),
                )
                reason = "evidence_quota_full"
                saved, lease = self._pause_for_evidence_quota(
                    tx, snapshot, now, reason=reason, preserve_closeup_wait=is_closeup
                )
                if lease:
                    release_after.append((lease, reason))
                self._replace_event_snapshot(tx, saved)
                raise _CommandError(
                    reason,
                    "Evidence storage is at its configured limit; free space or remove eligible evidence, then retry the requested close-up."
                    if is_closeup else
                    "Evidence storage is at its configured limit; free space or remove eligible evidence, then explicitly resume.",
                    result={"snapshot": saved, "execution_outcome": "waiting_evidence" if is_closeup else "paused"},
                )
            except QuotaExceeded:
                reason = "evidence_quota_full"
                saved, lease = self._pause_for_evidence_quota(
                    tx, snapshot, now, reason=reason, preserve_closeup_wait=is_closeup
                )
                if lease:
                    release_after.append((lease, reason))
                self._replace_event_snapshot(tx, saved)
                raise _CommandError(
                    reason,
                    "Evidence storage is at its configured limit; free space or remove eligible evidence, then retry the requested close-up."
                    if is_closeup else
                    "Evidence storage is at its configured limit; free space or remove eligible evidence, then explicitly resume.",
                    result={"snapshot": saved, "execution_outcome": "waiting_evidence" if is_closeup else "paused"},
                )
            updated = dict(snapshot)
            updated.setdefault("evidence", []).append(evidence)
            updated["input_evidence_id"] = evidence["evidence_id"]
            should_resume = bool(is_closeup)
            if should_resume:
                updated["source_binding"] = asdict(binding) if binding else updated.get("source_binding")
                updated["closeup_request"] = None
                updated["execution_generation"] = int(updated.get("execution_generation", 0)) + 1
                updated["cycle_id"] = uuid.uuid4().hex
                blockers = tx.list_running_missions(excluding=mission_id)
                other_draining = sorted(preflight.get("draining_missions", ()))
                if blockers or other_draining:
                    updated["state"] = "paused"
                    updated["reason"] = "inference_busy"
                    updated["activity"] = "Close-up received; another mission is active. Resume this inspection after it stops."
                    updated["cycle_id"] = None
                else:
                    updated["state"] = "running"
                    updated["reason"] = None
                    updated["activity"] = "Close-up received; starting a fresh inspection cycle."
                    followup.append((mission_id, updated["execution_generation"]))
            elif snapshot.get("state") == "running":
                old_lease = updated.get("watch_lease")
                updated["watch_lease"] = None
                updated["state"] = "paused"
                updated["reason"] = "replacement_evidence_attached"
                updated["execution_generation"] = int(updated.get("execution_generation", 0)) + 1
                updated["activity"] = "New evidence attached; the previous generation was invalidated. Resume explicitly to inspect it."
                if old_lease:
                    release_after.append((old_lease, "replacement_evidence_attached"))
            updated = self._update_with_event(tx, snapshot, updated, "evidence_attached", {"evidence": self._public_evidence(evidence), "snapshot": None}, now)
            self._replace_event_snapshot(tx, updated)
            return {"snapshot": updated, "evidence_id": evidence["evidence_id"], "sha256": evidence["sha256"], "width": evidence["width"], "height": evidence["height"]}
        if name == "activate":
            snapshot = self._require_revision(tx, mission_id, command)
            if preflight.get("same_mission_busy"):
                raise _CommandError("runtime_busy", "the previous generation is still draining; retry activate after it finishes", result={"snapshot": snapshot, "execution_outcome": "waiting_for_source"})
            if snapshot.get("state") not in {"created", "paused"} or snapshot.get("activation_intent") != "when_source_starts":
                raise _CommandError("invalid_state", "mission is not prepared for activation", result={"snapshot": snapshot})
            if set(args) != {"source_binding"}:
                raise _CommandError("source_binding_required", "activate requires the exact current source_binding", result={"snapshot": snapshot, "execution_outcome": "waiting_for_source"})
            binding = _source_binding(args.get("source_binding"))
            if (
                not binding.source_epoch.isascii()
                or not binding.source_epoch.isdigit()
                or str(int(binding.source_epoch)) != binding.source_epoch
            ):
                raise _CommandError("invalid_request", "source_epoch must be a canonical decimal string")
            prepared_binding = snapshot.get("source_binding") or {}
            if (
                binding.source_id != prepared_binding.get("source_id")
                or binding.source_id not in self.approved_source_ids
            ):
                raise _CommandError("source_binding_mismatch", "source_binding does not match the prepared approved source_id", result={"snapshot": snapshot, "execution_outcome": "waiting_for_source"})
            frame = self.source_provider(binding)
            if not self._frame_matches(frame, binding, max_age=2.0):
                raise _CommandError("source_unavailable", "the exact source epoch has no fresh accepted frame", result={"snapshot": snapshot, "execution_outcome": "waiting_for_source"})
            running = tx.list_running_missions(excluding=mission_id)
            blockers = sorted({item["mission_id"] for item in running} | set(preflight.get("draining_missions", ())))
            if blockers:
                raise _CommandError("inference_busy", "another mission owns the visual inference slot", result={"snapshot": snapshot, "blocking_mission_id": blockers[0], "execution_outcome": "paused"})
            model, mode = snapshot.get("reasoning_model"), snapshot.get("mode")
            if mode not in self.qualified_models.get(model, frozenset()) and mode not in self.evaluation_overrides.get(model, frozenset()):
                raise _CommandError("model_unqualified", "selected model/mode pair has neither measured qualification nor an operator evaluation override", result={"snapshot": snapshot, "execution_outcome": "paused"})
            if not self._model_status(model)[0]:
                raise _CommandError("model_unavailable", "selected model is not currently installed and ready", result={"snapshot": snapshot, "execution_outcome": "paused"})
            if mode == MODE_INSPECT:
                evidence_id = snapshot.get("input_evidence_id")
                if not evidence_id:
                    raise _CommandError("evidence_required", "Inspect needs attached evidence before activation", result={"snapshot": snapshot, "execution_outcome": "paused"})
                self._verified_evidence(evidence_id, mission_id, tx=tx)
            now_binding = self.source_provider(binding)
            if not self._frame_matches(now_binding, binding, max_age=2.0):
                raise _CommandError("source_unavailable", "source changed before activation committed", result={"snapshot": snapshot, "execution_outcome": "waiting_for_source"})
            updated = dict(snapshot)
            updated["source_binding"] = asdict(binding)
            updated["activation_intent"] = None
            updated["state"], updated["reason"] = "running", None
            updated["execution_generation"] = int(snapshot.get("execution_generation", 0)) + 1
            updated["cycle_id"] = uuid.uuid4().hex
            updated["configuration_revision"] = int(preflight.get("configuration_revision", snapshot.get("configuration_revision", 0)))
            updated["budget"] = self._budget_for_resume(snapshot.get("budget"), now)
            updated["model_provenance"] = self._model_provenance(model, mode)
            updated["activity"] = "Prepared mission activated on the exact approved source epoch."
            updated = self._update_with_event(tx, snapshot, updated, "mission_activated", {"source_binding": asdict(binding), "snapshot": None}, now)
            self._replace_event_snapshot(tx, updated)
            followup.append((mission_id, updated["execution_generation"]))
            return {"snapshot": updated, "brief_version": updated["brief_version"], "execution_outcome": "planning", "model_provenance": updated["model_provenance"]}
        if name == "update_brief":
            snapshot = self._require_revision(tx, mission_id, command)
            if set(args) - {"expertise", "goal", "watch_task"} or not args:
                raise _CommandError("invalid_request", "update_brief requires expertise, goal, watch_task, or a combination")
            watch_task = _watch_task(args)
            if watch_task is not None and snapshot.get("mode") != MODE_WATCH:
                raise _CommandError("invalid_request", "watch_task applies only to Watch missions")
            watch_task_changed = watch_task is not None and watch_task != snapshot.get("watch_task")
            expertise = (
                _clean_text(args["expertise"], limit=MAX_EXPERTISE_CHARS, field="expertise")
                if "expertise" in args else str(snapshot.get("expertise", ""))
            )
            goal = (
                _clean_text(args["goal"], limit=MAX_GOAL_CHARS, field="goal")
                if "goal" in args else str(snapshot.get("goal", ""))
            )
            if expertise == snapshot.get("expertise") and goal == snapshot.get("goal"):
                if not watch_task_changed:
                    return {"snapshot": snapshot, "brief_version": int(snapshot.get("brief_version", 1)), "execution_outcome": "unchanged"}
                updated = dict(snapshot)
                updated["watch_task"] = watch_task
                lease_data = snapshot.get("watch_lease")
                if (
                    snapshot.get("state") == "running"
                    and isinstance(lease_data, Mapping)
                    and lease_data.get("task") != watch_task
                ):
                    binding = self._binding_from_snapshot(snapshot)
                    lease_binding = _source_binding(lease_data.get("source_binding"))
                    if binding is None or binding != lease_binding:
                        raise _CommandError("source_binding_mismatch", "active Watch lease does not match the mission source", result={"snapshot": snapshot})
                    if int(lease_data.get("expires_at_ms", 0)) <= self.clock.now_ms():
                        raise _CommandError("watch_lease_expired", "active Watch lease expired before the task change", result={"snapshot": snapshot})
                    lease_revision = int(lease_data["configuration_revision"])
                    configuration_revision = self._configuration_revision()
                    if configuration_revision > lease_revision:
                        raise _CommandError("configuration_revision_conflict", "a newer configuration owns the source", result={"snapshot": snapshot})
                    request = WatchLeaseRequest(
                        mission_id=mission_id,
                        mission_revision=int(snapshot["revision"]),
                        execution_generation=int(snapshot["execution_generation"]),
                        source_binding=binding,
                        expected_configuration_revision=max(lease_revision, configuration_revision),
                        targets=tuple(lease_data["targets"]),
                        task=watch_task,
                        expires_at_ms=int(lease_data["expires_at_ms"]),
                    )
                    try:
                        lease = self.watch_adapter.apply(request)
                    except Exception as exc:
                        raise _CommandError("watch_lease_rejected", "active Watch lease did not accept the task change", result={"snapshot": snapshot}) from exc
                    if isinstance(lease, WatchLease):
                        applied_watch_leases.append(lease)
                    if (
                        lease.lease_id != lease_data.get("lease_id")
                        or lease.mission_id != mission_id
                        or lease.source_binding != binding
                        or tuple(lease.targets) != request.targets
                        or lease.task != watch_task
                        or lease.expires_at_ms != request.expires_at_ms
                    ):
                        raise _CommandError("watch_lease_invalid", "Watch task update changed the lease identity or expiry", result={"snapshot": snapshot})
                    updated["watch_lease"] = asdict(lease)
                    updated["configuration_revision"] = lease.configuration_revision
                    updated["targets"] = list(lease.targets)
                    updated["task"] = lease.task
                updated = self._update_with_event(
                    tx,
                    snapshot,
                    updated,
                    "watch_task_updated",
                    {
                        "watch_task": watch_task,
                        "task": updated.get("task", "detect"),
                        "lease_id": (updated.get("watch_lease") or {}).get("lease_id"),
                        "expires_at_ms": (updated.get("watch_lease") or {}).get("expires_at_ms"),
                        "snapshot": None,
                    },
                    now,
                )
                self._replace_event_snapshot(tx, updated)
                return {"snapshot": updated, "brief_version": int(updated.get("brief_version", 1)), "execution_outcome": "unchanged"}
            history = [dict(entry) for entry in snapshot.get("brief_history", []) if isinstance(entry, Mapping)]
            if not history:
                history = [_brief_history_entry(int(snapshot.get("brief_version", 1)), str(snapshot.get("expertise", "")), str(snapshot.get("goal", "")), int(snapshot.get("updated_at_ms", now)))]
            if len(history) >= MAX_BRIEF_VERSIONS:
                raise _CommandError("brief_history_full", "brief history reached its bounded 64-version limit", result={"snapshot": snapshot})
            version = int(snapshot.get("brief_version", 1)) + 1
            history.append(_brief_history_entry(version, expertise, goal, now))
            for entry in history[:-MAX_BRIEF_FULL_TEXT_VERSIONS]:
                entry.pop("expertise", None)
                entry.pop("goal", None)
            updated = dict(snapshot)
            old_lease = updated.get("watch_lease")
            updated["expertise"], updated["goal"] = expertise, goal
            if watch_task is not None:
                updated["watch_task"] = watch_task
            updated["brief_version"] = version
            updated["brief_sha256"] = _brief_sha256(expertise, goal)
            updated["brief_history"] = history[-MAX_BRIEF_VERSIONS:]
            updated["model_provenance"] = self._model_provenance(updated["reasoning_model"], updated["mode"])
            updated["execution_generation"] = int(snapshot.get("execution_generation", 0)) + 1
            updated["cycle_id"] = None
            updated["closeup_request"] = None
            updated["watch_lease"] = None
            updated["targets"] = []
            updated["task"] = "detect"
            outcome = "paused"
            if (
                snapshot.get("state") == "created"
                and snapshot.get("activation_intent") == "when_source_starts"
            ):
                updated["state"], updated["reason"] = "created", None
                updated["activity"] = "Brief updated; mission remains prepared until its source is explicitly activated."
                outcome = "waiting_for_source"
            elif snapshot.get("state") == "running" and snapshot.get("mode") == MODE_WATCH:
                binding = self._binding_from_snapshot(snapshot)
                frame = preflight.get("source_frame")
                if (
                    binding is not None
                    and binding.source_id in self.approved_source_ids
                    and self._frame_matches(frame, binding, max_age=2.0)
                ):
                    current_frame = self.source_provider(binding)
                    if self._frame_matches(current_frame, binding, max_age=2.0):
                        updated["state"], updated["reason"] = "running", None
                        updated["cycle_id"] = uuid.uuid4().hex
                        configuration_revision = int(preflight.get("configuration_revision", snapshot.get("configuration_revision", 0)))
                        if old_lease:
                            configuration_revision = max(
                                configuration_revision,
                                int(old_lease.get("configuration_revision", configuration_revision)),
                            ) + 1
                        updated["configuration_revision"] = configuration_revision
                        updated["budget"] = self._budget_for_resume(snapshot.get("budget"), now)
                        updated["activity"] = "Brief updated; planning a fresh cycle from the current approved source frame."
                        outcome = "planning"
                        followup.append((mission_id, updated["execution_generation"]))
                    else:
                        updated["state"], updated["reason"] = "paused", "source_unavailable"
                        updated["activity"] = "Brief updated; waiting for a fresh frame from the exact approved source epoch."
                        outcome = "paused"
                else:
                    updated["state"], updated["reason"] = "paused", "source_unavailable"
                    updated["activity"] = "Brief updated; waiting for a fresh frame from the exact approved source epoch."
                    outcome = "paused"
            elif snapshot.get("state") == "running" and snapshot.get("mode") == MODE_INSPECT:
                evidence_id = snapshot.get("input_evidence_id")
                try:
                    if evidence_id:
                        self._verified_evidence(evidence_id, mission_id, tx=tx)
                        updated["state"], updated["reason"] = "running", None
                        updated["cycle_id"] = uuid.uuid4().hex
                        updated["activity"] = "Brief updated; planning a fresh inspection cycle."
                        outcome = "planning"
                        followup.append((mission_id, updated["execution_generation"]))
                    else:
                        raise EvidenceUnavailable("input evidence missing")
                except EvidenceUnavailable:
                    updated["state"], updated["reason"] = "paused", "evidence_unavailable"
                    updated["activity"] = "Brief updated; attach fresh evidence before resuming."
            else:
                updated["activity"] = "Brief updated; mission remains paused until explicitly resumed."
            if old_lease:
                release_after.append((old_lease, "brief_updated"))
            updated = self._update_with_event(tx, snapshot, updated, "brief_updated", {"brief_version": version, "brief_sha256": updated["brief_sha256"], "execution_outcome": outcome, "snapshot": None}, now)
            self._replace_event_snapshot(tx, updated)
            return {"snapshot": updated, "brief_version": version, "execution_outcome": outcome, "brief_sha256": updated["brief_sha256"]}
        if name == "resume":
            snapshot = self._require_revision(tx, mission_id, command)
            if preflight.get("same_mission_busy"):
                raise _CommandError("runtime_busy", "the previous generation is still draining; retry resume after it finishes", result={"snapshot": snapshot})
            state = snapshot.get("state")
            if state not in {"created", "paused", "waiting_evidence"}:
                raise _CommandError("invalid_state", "mission is not resumable", result={"snapshot": snapshot})
            if snapshot.get("closeup_request") is not None:
                raise _CommandError("evidence_required", "provide the requested close-up or decline it before resuming", result={"snapshot": snapshot})
            running = tx.list_running_missions(excluding=mission_id)
            blockers = sorted({item["mission_id"] for item in running} | set(preflight.get("draining_missions", ())))
            if blockers:
                raise _CommandError("inference_busy", "another mission owns the visual inference slot", result={"snapshot": snapshot, "blocking_mission_id": blockers[0]})
            model = snapshot.get("reasoning_model")
            mode = snapshot.get("mode")
            if (
                mode not in self.qualified_models.get(model, frozenset())
                and mode not in self.evaluation_overrides.get(model, frozenset())
            ):
                raise _CommandError("model_unqualified", "selected model/mode pair has neither measured qualification nor an operator evaluation override")
            if not preflight.get("planner_ready", False):
                raise _CommandError("model_unavailable", "selected model is not currently installed and ready")
            binding = None
            if mode == MODE_INSPECT:
                evidence_id = snapshot.get("input_evidence_id")
                if not evidence_id:
                    raise _CommandError("evidence_required", "Inspect needs an attached photo before resume")
                try:
                    self._verified_evidence(evidence_id, mission_id, tx=tx)
                except EvidenceUnavailable:
                    raise _CommandError("evidence_unavailable", "original evidence is missing or changed")
            else:
                if set(args) != {"source_binding"}:
                    raise _CommandError("source_rebind_required", "Watch resume requires an explicit current source_binding")
                binding = _source_binding(args.get("source_binding"))
                frame = preflight.get("source_frame")
                if not self._frame_matches(frame, binding, max_age=2.0):
                    raise _CommandError("source_unavailable", "bound source epoch has no fresh approved frame")
                configuration_revision = int(preflight.get("configuration_revision", -1))
                if configuration_revision < 0:
                    raise _CommandError("control_state_unavailable", "bridge configuration revision is invalid")
            updated = dict(snapshot)
            if binding is not None:
                updated["source_binding"] = asdict(binding)
                updated["configuration_revision"] = configuration_revision
            updated["state"] = "running"
            updated["reason"] = None
            updated["model_provenance"] = self._model_provenance(model, mode)
            updated["execution_generation"] = int(snapshot.get("execution_generation", 0)) + 1
            updated["cycle_id"] = uuid.uuid4().hex
            updated["budget"] = self._budget_for_resume(snapshot.get("budget"), now)
            updated["activity"] = "Mission resumed by an authorized principal."
            updated["closeup_request"] = None if state == "waiting_evidence" else updated.get("closeup_request")
            updated = self._update_with_event(tx, snapshot, updated, "mission_resumed", {"snapshot": None}, now)
            self._replace_event_snapshot(tx, updated)
            followup.append((mission_id, updated["execution_generation"]))
            return {"snapshot": updated}
        if name in {"pause", "cancel", "decline_closeup"}:
            snapshot = self._require_revision(tx, mission_id, command)
            if name == "decline_closeup" and snapshot.get("closeup_request") is None:
                raise _CommandError("invalid_state", "mission has no pending close-up request")
            if snapshot.get("state") in {"completed", "cancelled", "failed"}:
                raise _CommandError("invalid_state", "terminal mission cannot change state", result={"snapshot": snapshot})
            updated = dict(snapshot)
            old_lease = updated.get("watch_lease")
            updated["watch_lease"] = None
            updated["execution_generation"] = int(snapshot.get("execution_generation", 0)) + 1
            if name == "cancel":
                updated["state"], updated["reason"] = "cancelled", "operator_cancelled"
                updated["closeup_request"] = None
                kind, activity = "mission_cancelled", "Mission cancelled by an authorized principal."
            elif name == "decline_closeup":
                updated["state"], updated["reason"] = "paused", "closeup_declined"
                updated["closeup_request"] = None
                kind, activity = "closeup_declined", "Close-up declined; mission paused unresolved."
            else:
                updated["state"], updated["reason"] = "paused", "operator_paused"
                updated["closeup_request"] = None
                kind, activity = "mission_paused", "Mission paused by an authorized principal."
            updated["activity"] = activity
            updated = self._update_with_event(tx, snapshot, updated, kind, {"snapshot": None}, now)
            self._replace_event_snapshot(tx, updated)
            if old_lease:
                release_after.append((old_lease, updated["reason"]))
            return {"snapshot": updated}
        if name == "review_finding":
            snapshot = self._require_revision(tx, mission_id, command)
            if set(args) - {"finding_id", "decision", "note"}:
                raise _CommandError("invalid_request", "review fields are invalid")
            finding_id = args.get("finding_id")
            decision = args.get("decision")
            if not isinstance(finding_id, str) or decision not in {"accepted", "corrected", "rejected"}:
                raise _CommandError("invalid_request", "finding review is malformed")
            note = _clean_text(args.get("note", ""), limit=1_000, field="note", required=False)
            updated = dict(snapshot)
            findings = [dict(item) for item in updated.get("findings", [])]
            finding = next((item for item in findings if item.get("finding_id") == finding_id), None)
            if finding is None:
                raise _CommandError("finding_not_found", "finding does not exist")
            if finding.get("review") is not None:
                raise _CommandError("finding_already_reviewed", "finding has already been reviewed")
            finding["review"] = {
                "decision": decision,
                "actor": principal.principal_id,
                "note": note,
                "time_ms": now,
                "mission_revision": int(snapshot["revision"]) + 1,
            }
            updated["findings"] = findings
            updated = self._update_with_event(tx, snapshot, updated, "finding_reviewed", {"finding_id": finding_id, "snapshot": None}, now)
            self._replace_event_snapshot(tx, updated)
            return {"snapshot": updated}
        raise _CommandError("invalid_request", "command is not implemented")

    def _require_revision(self, tx, mission_id: str, command: Mapping[str, Any]) -> dict[str, Any]:
        snapshot = tx.get_mission(mission_id)
        if snapshot is None:
            raise KeyError(mission_id)
        if int(snapshot["revision"]) != int(command["expected_revision"]):
            raise RevisionConflict(snapshot)
        return snapshot

    def _update_with_event(self, tx, before: Mapping[str, Any], updated: Mapping[str, Any], kind: str, data: Mapping[str, Any], now: int) -> dict[str, Any]:
        saved = tx.update_mission(updated, expected_revision=int(before["revision"]), updated_at_ms=now)
        saved["last_sequence"] = int(saved.get("last_sequence", 0)) + 1
        event_data = dict(data)
        event_data["snapshot"] = self._event_snapshot(saved)
        tx.append_event(saved["mission_id"], saved["revision"], kind, event_data, now)
        return tx.get_mission(saved["mission_id"])

    @staticmethod
    def _mission_summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        fields = (
            "mission_id", "revision", "state", "reason", "expertise", "goal", "mode",
            "reasoning_model", "source_binding", "input_evidence_id", "activity", "targets",
            "task", "closeup_request", "budget", "cycle_id", "last_sequence", "updated_at_ms",
        )
        return {key: snapshot[key] for key in fields if key in snapshot} | {"findings": [], "evidence": []}

    @staticmethod
    def _event_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        # State arrays have explicit mission caps; this also bounds events for a
        # pre-cap legacy snapshot recovered from disk.
        event_snapshot = dict(snapshot)
        event_snapshot["evidence"] = list(snapshot.get("evidence", []))[-MAX_MISSION_EVIDENCE_HISTORY:]
        event_snapshot["findings"] = list(snapshot.get("findings", []))[-MAX_MISSION_FINDINGS:]
        return event_snapshot

    def _replace_event_snapshot(self, tx, snapshot: Mapping[str, Any]) -> None:
        # append_event runs after the state update; the returned transaction snapshot
        # carries the committed sequence cursor used by later pages.
        current = tx.get_mission(snapshot["mission_id"])
        if current and current.get("last_sequence") != snapshot.get("last_sequence"):
            return

    def _reply(self, command: Mapping[str, Any], ok: bool, result: Mapping[str, Any] | None = None, error: tuple[str, str] | None = None) -> dict[str, Any]:
        error_obj = {"code": error[0], "message": error[1]} if error else None
        reply = {
            "type": "mission_reply",
            "schema_version": MISSION_SCHEMA_VERSION,
            "request_id": command.get("request_id"),
            "mission_id": command.get("mission_id"),
            "revision": (
                (result or {}).get("snapshot", {}).get("revision")
                if isinstance((result or {}).get("snapshot"), Mapping)
                else (result or {}).get("mission_revision")
            ),
            "ok": bool(ok),
            "result": dict(result or {}),
            "error": error_obj,
        }
        try:
            size = len(json.dumps(reply, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError):
            size = MAX_OUTBOUND_MESSAGE_BYTES + 1
        if size > MAX_OUTBOUND_MESSAGE_BYTES:
            mutation = command.get("command") in {
                "create", "prepare", "activate", "update_brief", "attach_evidence", "resume", "pause", "cancel",
                "review_finding", "decline_closeup",
            }
            current_snapshot = (result or {}).get("snapshot")
            if not isinstance(current_snapshot, Mapping):
                packet = (result or {}).get("packet")
                current_snapshot = packet.get("mission") if isinstance(packet, Mapping) else None
            if isinstance(current_snapshot, Mapping):
                reply["mission_id"] = current_snapshot.get("mission_id", reply["mission_id"])
                reply["revision"] = current_snapshot.get("revision")
            if command.get("command") == "preview_packet":
                reply["result"] = {
                    "recovery": "get",
                    "mission_id": reply["mission_id"],
                    "revision": reply["revision"],
                }
            else:
                reply["result"] = {}
            reply["ok"] = False
            reply["error"] = {
                "code": "result_too_large",
                "message": "mission response exceeds 256 KiB; read the authoritative mission snapshot",
                "retryable": command.get("command") != "preview_packet",
                "outcome_unknown": bool(mutation),
            }
        return reply

    def _publish_events(self, events: Collection[Mapping[str, Any]]) -> None:
        for event in events:
            try:
                self.event_sink(MissionEvent(
                    mission_id=event["mission_id"],
                    revision=event["revision"],
                    sequence=event["sequence"],
                    kind=event["kind"],
                    data=event["data"],
                    timestamp_ms=event["timestamp_ms"],
                ))
            except Exception:
                # Durable state already committed; a disconnected event subscriber
                # can recover through events_since without changing that state.
                continue

    async def _native_call(self, function, *args, mission_id: str | None = None, deadline: float | None = None, on_deadline=None):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        self._native_calls.add(task)
        task.add_done_callback(self._native_calls.discard)
        try:
            timeout = None if deadline is None else max(0.0, deadline - self.clock.monotonic())
            done, _pending = await asyncio.wait({task}, timeout=timeout)
            if not done:
                if on_deadline is not None and not self._closed:
                    await on_deadline()
                try:
                    await asyncio.shield(task)
                except Exception:
                    pass
                raise _NativeDeadlineExpired("native call exceeded the active mission deadline")
            return task.result()
        except asyncio.CancelledError:
            # to_thread cannot stop native work. Drain it before this caller exits.
            await asyncio.shield(task)
            raise

    def _start_task(self, mission_id: str, generation: int) -> None:
        prior = self._tasks.get(mission_id)
        if prior is not None and not prior.done():
            self._pending_generations[mission_id] = generation
            return
        if any(key != mission_id and not task.done() for key, task in self._tasks.items()):
            self._pending_generations[mission_id] = generation
            return
        self._task_interrupts[mission_id] = asyncio.Event()
        task = asyncio.create_task(self._run_mission(mission_id, generation), name=f"mission-{mission_id}")
        self._tasks[mission_id] = task
        task.add_done_callback(lambda done, key=mission_id: self._task_finished(key, done))

    def _task_finished(self, mission_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(mission_id) is task:
            self._tasks.pop(mission_id, None)
            self._task_interrupts.pop(mission_id, None)
        if self._pending_generations and not self._closed:
            asyncio.create_task(self._launch_pending())

    def _interrupt_watch_pacing(self, mission_id: str) -> None:
        interrupt = self._task_interrupts.get(mission_id)
        if interrupt is not None:
            interrupt.set()

    async def _wait_for_watch_interval(self, mission_id: str) -> None:
        interrupt = self._task_interrupts.get(mission_id)
        if interrupt is None:
            await asyncio.sleep(WATCH_MIN_INTERVAL_SECONDS)
            return
        try:
            await asyncio.wait_for(interrupt.wait(), timeout=WATCH_MIN_INTERVAL_SECONDS)
        except asyncio.TimeoutError:
            pass

    async def _launch_pending(self) -> None:
        if any(not task.done() for task in self._tasks.values()):
            return
        for mission_id, generation in tuple(self._pending_generations.items()):
            snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
            if not self._is_current(snapshot, generation, running=True):
                self._pending_generations.pop(mission_id, None)
                continue
            self._pending_generations.pop(mission_id, None)
            self._start_task(mission_id, generation)
            return

    async def _run_mission(self, mission_id: str, generation: int) -> None:
        last_frame_key: tuple[str, str, int] | None = None
        try:
            while not self._closed:
                snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
                if not self._is_current(snapshot, generation, running=True):
                    return
                mode = snapshot["mode"]
                if mode == MODE_INSPECT:
                    evidence_id = snapshot.get("input_evidence_id")
                    if not evidence_id:
                        await self._set_waiting(mission_id, generation, "evidence_required", "Attach a photo before Inspect can continue.")
                        return
                    try:
                        evidence = await asyncio.to_thread(self.store.read_evidence, evidence_id)
                    except EvidenceUnavailable as exc:
                        await asyncio.to_thread(
                            self._mark_evidence_unavailable,
                            mission_id,
                            evidence_id,
                            exc.availability_reason,
                        )
                        await self._set_waiting(mission_id, generation, "evidence_unavailable", "The original photo is missing or changed; attach it again.")
                        return
                    except KeyError:
                        await asyncio.to_thread(
                            self._mark_evidence_unavailable, mission_id, evidence_id, "missing"
                        )
                        await self._set_waiting(mission_id, generation, "evidence_unavailable", "The original photo is missing or changed; attach it again.")
                        return
                    if self._closed:
                        return
                    metadata = evidence["metadata"]
                    binding = self._binding_from_snapshot(snapshot)
                    await self._run_cycle(
                        mission_id,
                        generation,
                        evidence_id,
                        evidence["jpeg_bytes"],
                        metadata["width"],
                        metadata["height"],
                        binding,
                        metadata.get("frame_id"),
                        _transform(metadata.get("input_transform") or None),
                        None,
                    )
                    return

                binding = self._binding_from_snapshot(snapshot)
                if binding is None:
                    await self._set_waiting(mission_id, generation, "source_rebind_required", "Watch needs an explicit approved source binding.")
                    return
                frame = self.source_provider(binding)
                if not self._frame_matches(frame, binding, max_age=2.0):
                    await self._set_waiting(mission_id, generation, "source_unavailable", "The approved source has no fresh frame; resume after reconnect and rebind.")
                    return
                key = (frame.source_id, frame.source_epoch, frame.frame_id)
                if key == last_frame_key:
                    await self._wait_for_watch_interval(mission_id)
                    continue
                evidence_id, meta = await asyncio.to_thread(self._save_watch_frame, mission_id, generation, frame)
                if self._closed:
                    return
                if not evidence_id:
                    if meta.get("watch_lease"):
                        await self._release_lease(meta["watch_lease"], meta.get("release_reason", "evidence_quota_full"))
                    return
                last_frame_key = key
                await self._run_cycle(
                    mission_id,
                    generation,
                    evidence_id,
                    frame.jpeg_bytes,
                    meta["width"],
                    meta["height"],
                    binding,
                    frame.frame_id,
                    None,
                    frame.received_monotonic,
                )
                if self._closed:
                    return
                snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
                if not self._is_current(snapshot, generation, running=True):
                    return
                # Watch gets a fresh latest-only frame at a paced interval. It never
                # queues old frames or extends the one-cycle execution budget.
                await self._wait_for_watch_interval(mission_id)
        except Exception:
            if self._closed:
                return
            snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
            if self._is_current(snapshot, generation, running=True):
                await self._set_waiting(mission_id, generation, "runtime_error", "Mission stopped safely after an internal runtime error.", state="failed")

    async def _run_cycle(
        self,
        mission_id: str,
        generation: int,
        evidence_id: str,
        jpeg: bytes,
        width: int,
        height: int,
        binding: SourceBinding | None,
        frame_id: int | None,
        transform: InputTransform | None,
        frame_received_monotonic: float | None,
    ) -> None:
        digest = hashlib.sha256(jpeg).hexdigest()
        cycle_id = uuid.uuid4().hex
        records: list[ToolCallRecord] = []
        grounded: dict[str, GeometryItem] = {}
        used: set[tuple[str, str]] = set()
        repair: str | None = None
        repair_count = 0
        last_response: Any = None
        tool_count = 0
        deadline = self.clock.monotonic() + MAX_ACTIVE_SECONDS

        async def expire_deadline() -> None:
            await self._set_waiting(
                mission_id,
                generation,
                "active_budget_exhausted",
                "The active time budget expired; the current native operation is draining and will not be applied.",
                state="paused",
            )

        async def record_diagnostic(
            attempt: int,
            *,
            raw_text: str = "",
            error: str = "",
            prompt_version: str = "",
            checkpoint: str = "",
        ) -> None:
            recorder = getattr(self.store, "add_planner_diagnostic", None)
            if not callable(recorder):
                return
            current = await asyncio.to_thread(self.store.get_mission, mission_id)
            provenance = current.get("model_provenance", {}) if isinstance(current, Mapping) else {}
            try:
                await asyncio.to_thread(
                    recorder,
                    mission_id,
                    cycle_id=cycle_id,
                    execution_generation=generation,
                    attempt=attempt,
                    model_key=str(current.get("reasoning_model", "")) if isinstance(current, Mapping) else "",
                    checkpoint=checkpoint or (
                        str(provenance.get("checkpoint", ""))
                        if isinstance(provenance, Mapping) else ""
                    ),
                    prompt_version=prompt_version or str(getattr(self.planner, "prompt_version", "unknown")),
                    raw_text=raw_text,
                    error=error,
                    repair_feedback=repair,
                    created_at_ms=self.clock.now_ms(),
                )
            except Exception:
                # Diagnostics must never change the mission's execution outcome.
                return

        async def record_finish_diagnostic(error: str) -> None:
            await record_diagnostic(
                repair_count,
                raw_text=str(getattr(last_response, "raw_text", "")),
                error=error,
                prompt_version=str(getattr(last_response, "prompt_version", "")),
                checkpoint=str(getattr(last_response, "checkpoint", "")),
            )

        for generation_index in range(MAX_GENERATIONS_PER_CYCLE):
            if self.clock.monotonic() >= deadline:
                await expire_deadline()
                return
            if not await self._charge_budget(mission_id, generation, "generation"):
                await self._set_waiting(mission_id, generation, "rolling_generation_budget_exhausted", "The five-minute planner budget is exhausted; resume after the window resets.", state="paused")
                return
            if self._closed:
                return
            snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
            if not self._is_current(snapshot, generation, running=True):
                return
            allowed = self._allowed_tools(snapshot)
            if "finish" not in allowed:
                await self._set_waiting(mission_id, generation, "qualified_tools_unavailable", "No qualified perception tools are currently available.", state="paused")
                return
            context = PlannerContext(
                mission_id=mission_id,
                revision=int(snapshot["revision"]),
                cycle_id=cycle_id,
                execution_generation=generation,
                profile_id=PROFILE_ID,
                profile_version=PROFILE_VERSION,
                mode=snapshot["mode"],
                expertise=snapshot["expertise"],
                goal=snapshot["goal"],
                reasoning_model=snapshot["reasoning_model"],
                input_evidence_id=evidence_id,
                input_sha256=digest,
                image_jpeg=jpeg,
                image_width=width,
                image_height=height,
                source_binding=binding,
                frame_id=frame_id,
                input_transform=transform,
                allowed_tools=tuple(allowed.values()),
                tool_results=tuple(records),
                findings=tuple(snapshot.get("findings", [])),
                generations_remaining=int(snapshot.get("budget", {}).get("generations_remaining", 0)),
                tool_calls_remaining=min(
                    MAX_TOOL_CALLS_PER_CYCLE - tool_count,
                    int(snapshot.get("budget", {}).get("window_tools_remaining", MAX_TOOL_CALLS_PER_WINDOW)),
                ),
                repair_feedback=repair,
                brief_version=int(snapshot.get("brief_version", 1)),
                brief_sha256=snapshot.get("brief_sha256"),
                model_provenance=dict(snapshot.get("model_provenance", {})),
            )
            if self.clock.monotonic() >= deadline:
                await expire_deadline()
                return
            try:
                if callable(getattr(self.planner, "plan_with_response", None)):
                    response = await self._native_call(self.planner.plan_with_response, context, mission_id=mission_id, deadline=deadline, on_deadline=expire_deadline)
                    if self._closed:
                        return
                    if not getattr(response, "valid", False) or getattr(response, "decision", None) is None:
                        repair = str(getattr(response, "error", "invalid planner response"))[:500]
                        await record_diagnostic(
                            repair_count,
                            raw_text=str(getattr(response, "raw_text", "")),
                            error=repair,
                            prompt_version=str(getattr(response, "prompt_version", "")),
                            checkpoint=str(getattr(response, "checkpoint", "")),
                        )
                        if repair_count >= 1:
                            await self._set_waiting(mission_id, generation, "planner_repair_exhausted", "The single planner repair was invalid; mission paused unresolved.", state="paused")
                            return
                        repair_count += 1
                        continue
                    decision = response.decision
                    last_response = response
                else:
                    decision = await self._native_call(self.planner.plan, context, mission_id=mission_id, deadline=deadline, on_deadline=expire_deadline)
                    if self._closed:
                        return
                action, arguments = _validate_action(decision, allowed_tools=set(allowed), grounded_items=grounded)
            except _NativeDeadlineExpired:
                return
            except Exception as exc:
                if self._closed:
                    return
                repair = str(exc)[:500] or "planner output was invalid"
                await record_diagnostic(
                    repair_count,
                    raw_text=str(getattr(exc, "raw_text", "")),
                    error=repair,
                )
                if repair_count >= 1:
                    await self._set_waiting(mission_id, generation, "planner_repair_exhausted", "The single planner repair was invalid; mission paused unresolved.", state="paused")
                    return
                repair_count += 1
                continue
            if self.clock.monotonic() >= deadline:
                await expire_deadline()
                return
            if not self._is_current(await asyncio.to_thread(self.store.get_mission, mission_id), generation, running=True):
                return
            if action == "request_closeup":
                await self._request_closeup(mission_id, generation, evidence_id, decision, arguments)
                return
            if action == "finish":
                if snapshot["mode"] == MODE_WATCH and decision.watch is not None and (not decision.watch.targets or len(decision.watch.targets) > MAX_TARGETS):
                    repair = "Watch finish requires at least one bounded target."
                    await record_finish_diagnostic(repair)
                    if repair_count >= 1:
                        await self._set_waiting(mission_id, generation, "planner_repair_exhausted", "The single planner repair was invalid; mission paused unresolved.", state="paused")
                        return
                    repair_count += 1
                    continue
                try:
                    findings = self._findings(
                        decision,
                        records,
                        evidence_id,
                        binding,
                        frame_id,
                        brief_version=int(snapshot.get("brief_version", 1)),
                        brief_sha256=snapshot.get("brief_sha256"),
                        model_provenance=snapshot.get("model_provenance", {}),
                    )
                except ValueError as exc:
                    repair = str(exc)[:500]
                    await record_finish_diagnostic(repair)
                    if repair_count >= 1:
                        await self._set_waiting(mission_id, generation, "planner_repair_exhausted", "The single planner repair was invalid; mission paused unresolved.", state="paused")
                        return
                    repair_count += 1
                    continue
                if snapshot["mode"] == MODE_WATCH:
                    watch_proposal = decision.watch
                    if watch_proposal is not None:
                        if self.clock.monotonic() >= deadline or not self._is_current(await asyncio.to_thread(self.store.get_mission, mission_id), generation, running=True):
                            await expire_deadline()
                            return
                        await self._apply_watch(
                            mission_id,
                            generation,
                            binding,
                            watch_proposal,
                            frame_received_monotonic,
                        )
                if self.clock.monotonic() >= deadline or not self._is_current(await asyncio.to_thread(self.store.get_mission, mission_id), generation, running=True):
                    await expire_deadline()
                    return
                await self._finish(mission_id, generation, cycle_id, findings, decision.watch)
                return
            action_key = (action, json.dumps(arguments, sort_keys=True, separators=(",", ":")))
            if action_key in used:
                repair = "This identical tool action already ran in this cycle; choose a different action or finish."
                await record_finish_diagnostic(repair)
                if repair_count >= 1:
                    await self._set_waiting(mission_id, generation, "planner_repair_exhausted", "The single planner repair was invalid; mission paused unresolved.", state="paused")
                    return
                repair_count += 1
                continue
            if tool_count >= MAX_TOOL_CALLS_PER_CYCLE:
                await self._set_waiting(mission_id, generation, "tool_budget_exhausted", "Tool-call budget expired.", state="paused")
                return
            used.add(action_key)
            tool_count += 1
            if not await self._charge_budget(mission_id, generation, "tool"):
                await self._set_waiting(mission_id, generation, "rolling_tool_budget_exhausted", "The five-minute perception-tool budget is exhausted; resume after the window resets.", state="paused")
                return
            request = ToolRequest(
                tool=action,
                arguments=arguments,
                mission_id=mission_id,
                cycle_id=cycle_id,
                execution_generation=generation,
                reasoning_model=snapshot["reasoning_model"],
                input_evidence_id=evidence_id,
                input_sha256=digest,
                frame_id=frame_id,
                source_binding=binding,
                brief_version=int(snapshot.get("brief_version", 1)),
                brief_sha256=snapshot.get("brief_sha256"),
                model_provenance=dict(snapshot.get("model_provenance", {})),
            )
            tool_context = ToolContext(jpeg, width, height, dict(grounded), tuple(records), transform)
            if self.clock.monotonic() >= deadline:
                await expire_deadline()
                return
            try:
                result = await self._native_call(self.tools.execute, request, tool_context, mission_id=mission_id, deadline=deadline, on_deadline=expire_deadline)
                if self._closed:
                    return
                if not isinstance(result, ToolResult):
                    raise ValueError("tool adapter returned an untyped result")
            except _NativeDeadlineExpired:
                return
            except Exception:
                if self._closed:
                    return
                result = ToolResult(status="failed", error_code="native_inference_failed")
            if self.clock.monotonic() >= deadline:
                await expire_deadline()
                return
            record, quota_lease = await asyncio.to_thread(
                self._persist_tool_result,
                mission_id,
                generation,
                cycle_id,
                evidence_id,
                action,
                result,
                brief_version=request.brief_version,
                brief_sha256=request.brief_sha256,
                model_provenance=request.model_provenance,
                source_binding=request.source_binding,
                frame_id=request.frame_id,
                input_sha256=request.input_sha256,
            )
            if quota_lease:
                await self._release_lease(quota_lease, record.error_code or "evidence_quota_full")
            if self._closed:
                return
            records.append(record)
            for item in record.items:
                grounded[item.item_id] = item
            if not self._is_current(await asyncio.to_thread(self.store.get_mission, mission_id), generation, running=True):
                return
            repair = None
        await self._set_waiting(mission_id, generation, "planner_budget_exhausted", "Planner generation budget expired without a finish decision.", state="paused")

    def _allowed_tools(self, snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        from .mission_models import MISSION_TOOL_SCHEMAS

        available = set()
        try:
            available = set(self.tools.available_tools())
        except Exception:
            pass
        model = snapshot["reasoning_model"]
        mode = snapshot["mode"]
        if (
            mode not in self.qualified_models.get(model, frozenset())
            and mode not in self.evaluation_overrides.get(model, frozenset())
        ):
            return {}
        try:
            if not self.planner.available(model):
                return {}
        except Exception:
            return {}
        names = set(available & PERCEPTION_TOOL_NAMES) | {"finish", "request_closeup"}
        if mode == MODE_WATCH:
            names.intersection_update({"detect_objects", "segment_objects", "request_closeup", "finish"})
        return {schema["name"]: schema for schema in MISSION_TOOL_SCHEMAS if schema["name"] in names}

    async def _charge_budget(self, mission_id: str, generation: int, kind: str) -> bool:
        now = self.clock.now_ms()

        def operation(tx):
            snapshot = tx.get_mission(mission_id)
            if not self._is_current(snapshot, generation, running=True):
                return False
            budget = self._budget_for_dispatch(snapshot.get("budget"), now)
            if kind == "generation":
                used_key, remaining_key = "window_generations_used", "generations_remaining"
                window_limit = MAX_GENERATIONS_PER_WINDOW
            elif kind == "tool":
                used_key, remaining_key = "window_tools_used", "tools_remaining"
                window_limit = MAX_TOOL_CALLS_PER_WINDOW
            else:
                raise ValueError("unknown mission budget kind")
            if int(budget.get(remaining_key, 0)) <= 0 or int(budget.get(used_key, 0)) >= window_limit:
                return False
            budget[used_key] = int(budget.get(used_key, 0)) + 1
            budget[remaining_key] = max(0, int(budget.get(remaining_key, 0)) - 1)
            budget["window_generations_remaining"] = max(0, MAX_GENERATIONS_PER_WINDOW - int(budget.get("window_generations_used", 0)))
            budget["window_tools_remaining"] = max(0, MAX_TOOL_CALLS_PER_WINDOW - int(budget.get("window_tools_used", 0)))
            updated = dict(snapshot)
            updated["budget"] = budget
            updated["last_sequence"] = int(snapshot.get("last_sequence", 0)) + 1
            saved = tx.update_mission_activity(
                updated, expected_revision=int(snapshot["revision"])
            )
            tx.append_event(
                mission_id,
                int(snapshot["revision"]),
                "budget_updated",
                {"budget": budget, "snapshot": self._event_snapshot(saved)},
                now,
            )
            return True

        result = await asyncio.to_thread(self.store.transact, operation)
        self._publish_events(result.events)
        return bool(result.value)

    def _findings(
        self,
        decision: Decision,
        records: list[ToolCallRecord],
        evidence_id: str,
        binding: SourceBinding | None,
        frame_id: int | None,
        *,
        brief_version: int = 1,
        brief_sha256: str | None = None,
        model_provenance: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(decision.findings) > MAX_FINDINGS:
            raise ValueError("finish includes more than the finding limit")
        by_result = {record.tool_result_id: record for record in records if record.status == "ok" and record.input_evidence_id == evidence_id}
        output = []
        for proposal in decision.findings:
            if not isinstance(proposal, FindingProposal) or proposal.claim_type not in {"localized_object", "text_read", "visual_hypothesis"}:
                raise ValueError("finding proposal is malformed")
            if len(proposal.item_refs) > MAX_FINDING_ITEMS or len(proposal.text_refs) > MAX_FINDING_ITEMS:
                raise ValueError("finding includes more grounded references than the mission limit")
            claim = _clean_text(proposal.claim, limit=500, field="finding claim")
            evidence_refs = set(proposal.evidence_refs)
            evidence_refs.add(evidence_id) if not evidence_refs else None
            known_evidence = {evidence_id}
            for record in by_result.values():
                known_evidence.update(record.evidence_ids)
            if not evidence_refs.issubset(known_evidence):
                raise ValueError("finding references evidence outside the current cycle")
            items: list[tuple[ToolCallRecord, GeometryItem]] = []
            for result_id, item_id in proposal.item_refs:
                record = by_result.get(result_id)
                item = next((item for item in record.items if item.item_id == item_id), None) if record else None
                if item is None:
                    raise ValueError("finding references an item outside the current cycle")
                items.append((record, item))
            if proposal.claim_type == "localized_object" and not items:
                raise ValueError("localized_object finding requires a grounded item reference")
            status, reason = "unresolved", "visual_hypothesis_requires_review"
            localization = None
            if proposal.claim_type == "localized_object":
                # Geometry establishes where/count of tool-labeled detections; it
                # does not prove the planner's damage, safety, identity, or defect
                # claim. Keep that claim unresolved and author localization here.
                status, reason = "unresolved", "geometry_does_not_prove_semantic_claim"
                localization = self._server_localization(items, evidence_id)
            elif proposal.claim_type == "text_read":
                cited = [by_result.get(result_id) for result_id in proposal.text_refs]
                if not cited or any(record is None or record.tool != "read_text" or record.status != "ok" for record in cited):
                    raise ValueError("text_read finding must reference a current successful OCR result")
                if any(proposal.claim != record.text for record in cited):
                    status, reason = "unresolved", "claim_is_not_an_exact_ocr_quote"
                else:
                    status, reason = "supported", "exact_ocr_transcription_only"
            elif proposal.claim_type == "visual_hypothesis":
                status, reason = "unresolved", "visual_interpretation_is_a_hypothesis"
            # Tool geometry is normalized to the immutable input frame. Crop
            # artifacts may support context but must not become its coordinate origin.
            finding_evidence_id = evidence_id
            finding = {
                "finding_id": uuid.uuid4().hex,
                "claim": claim,
                "claim_type": proposal.claim_type,
                "status": status,
                "reason": reason,
                "evidence_id": finding_evidence_id,
                "evidence_refs": sorted(evidence_refs),
                "item_refs": [{"tool_result_id": result_id, "item_id": item_id} for result_id, item_id in proposal.item_refs],
                "items": [_geometry_dict(item) for _, item in items],
                "localization": localization,
                "source_binding": asdict(binding) if binding else None,
                "frame_id": frame_id,
                "brief_version": brief_version,
                "brief_sha256": brief_sha256,
                "model_provenance": dict(model_provenance or {}),
                "review": None,
            }
            if proposal.claim_type == "text_read":
                finding["source_kind"] = "ocr_untrusted_image_text"
                finding["text_refs"] = list(proposal.text_refs)
            output.append(finding)
        return output

    @staticmethod
    def _server_localization(items: list[tuple[ToolCallRecord, GeometryItem]], evidence_id: str) -> dict[str, Any]:
        by_label: dict[str, list[GeometryItem]] = {}
        for _record, item in items:
            by_label.setdefault(item.label, []).append(item)
        statements = []
        for label, grouped in sorted(by_label.items(), key=lambda pair: pair[0].casefold()):
            centers = [((item.box[0] + item.box[2]) / 2, (item.box[1] + item.box[3]) / 2) for item in grouped]
            positions = [{"x": round(x, 4), "y": round(y, 4)} for x, y in centers]
            statements.append({"claim": f"Detected {len(grouped)} {label} item(s) in the image.", "label": label, "count": len(grouped), "positions": positions})
        return {"status": "supported", "basis": "server_authored_tool_localization", "evidence_id": evidence_id, "statements": statements}

    async def _request_closeup(self, mission_id: str, generation: int, evidence_id: str, decision: Decision, arguments: Mapping[str, Any]) -> None:
        if self._closed:
            return
        now = self.clock.now_ms()
        snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
        if self._closed or not self._is_current(snapshot, generation, running=True):
            return
        request_id = uuid.uuid4().hex
        closeup = {
            "request_id": request_id,
            "description": arguments.get("description"),
            "reason": arguments["reason"],
            "item_id": arguments.get("item_id"),
            "evidence_id": evidence_id,
            "expires_at_ms": now + MAX_HUMAN_WAIT_SECONDS * 1000,
        }
        updated = dict(snapshot)
        lease = updated.get("watch_lease")
        updated["watch_lease"] = None
        updated["state"] = "waiting_evidence"
        updated["reason"] = "closeup_requested"
        updated["execution_generation"] = generation + 1
        updated["closeup_request"] = closeup
        updated["activity"] = "A closer photo is requested; mission is waiting up to five minutes."

        def operation(tx):
            if self._closed:
                return None
            return self._update_with_event(tx, snapshot, updated, "closeup_requested", {"closeup_request": closeup}, now)

        try:
            committed = await asyncio.to_thread(self.store.transact, operation)
        except RevisionConflict:
            return
        if committed.value is None:
            return
        self._publish_events(committed.events)
        if self._closed:
            if lease:
                await self._release_lease(lease, "runtime_shutdown")
            return
        if lease:
            await self._release_lease(lease, "closeup_requested")
        self._schedule_closeup_timeout(mission_id, request_id, closeup["expires_at_ms"])

    async def _finish(
        self,
        mission_id: str,
        generation: int,
        cycle_id: str,
        findings: list[dict[str, Any]],
        watch: WatchProposal | None,
    ) -> None:
        if self._closed:
            return
        snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
        if self._closed or not self._is_current(snapshot, generation, running=True):
            return
        now = self.clock.now_ms()
        merged_findings = list(snapshot.get("findings", []))
        new_findings = findings
        observation_updates: list[dict[str, Any]] = []
        if snapshot["mode"] == MODE_WATCH:
            merged_findings, new_findings, observation_updates = self._reconcile_watch_findings(
                merged_findings, findings, now
            )
            cycle_finding_ids = list(dict.fromkeys(
                update.get("finding_id")
                for update in observation_updates
                if isinstance(update.get("finding_id"), str)
            ))
        else:
            merged_findings.extend(findings)
            cycle_finding_ids = [item.get("finding_id") for item in findings]
        if len(merged_findings) > MAX_MISSION_FINDINGS:
            await self._set_waiting(mission_id, generation, "finding_budget_exhausted", "Finding history reached its bounded limit; review or export before creating another mission.", state="paused")
            return
        updated = dict(snapshot)
        updated["findings"] = merged_findings
        input_evidence_id = snapshot.get("input_evidence_id")
        primary_ref = next(
            (
                ref for ref in snapshot.get("evidence", ())
                if ref.get("evidence_id") == input_evidence_id
            ),
            None,
        )
        evidence_refs = [
            {
                "evidence_id": input_evidence_id,
                "available": bool(primary_ref.get("available", True)) if primary_ref else False,
                **(
                    {"availability_reason": primary_ref["availability_reason"]}
                    if primary_ref and primary_ref.get("availability_reason")
                    else {}
                ),
            }
        ] if input_evidence_id else []
        cycle_history = list(snapshot.get("cycle_history", ()))
        cycle_history.append(
            {
                "cycle_id": cycle_id,
                "execution_generation": generation,
                "brief_version": int(snapshot.get("brief_version", 1)),
                "brief_sha256": snapshot.get("brief_sha256"),
                "model_provenance": dict(snapshot.get("model_provenance", {})),
                "evidence_refs": evidence_refs,
                "finding_ids": cycle_finding_ids,
                "outcome": "completed",
                "finished_at_ms": now,
            }
        )
        updated["cycle_history"] = cycle_history[-MAX_MISSION_CYCLES:]
        if snapshot["mode"] == MODE_INSPECT:
            updated["state"], updated["reason"] = "completed", None
            updated["activity"] = "Review complete. Findings remain subject to operator review."
        else:
            updated["state"], updated["reason"] = "running", None
            updated["activity"] = "Watch cycle complete; waiting for the next fresh source frame."
            if watch is None and not snapshot.get("watch_lease"):
                updated["state"], updated["reason"] = "paused", "watch_lease_missing"
                updated["activity"] = "Watch has no active target lease; review and resume explicitly."
        updated["cycle_id"] = None
        event_data = {
            "new_findings": new_findings,
            "observation_updates": observation_updates,
        }
        if snapshot["mode"] == MODE_WATCH:
            # A committed cycle event is the Watch heartbeat even when the
            # scene is unchanged and no new finding row was created.
            event_data["watch_heartbeat"] = True
        updated = await self._change_snapshot(mission_id, generation, updated, "cycle_finished", event_data)
        if updated and updated["state"] in {"completed", "paused"}:
            old_lease = snapshot.get("watch_lease")
            if old_lease:
                await self._release_lease(old_lease, updated.get("reason") or "inspect_complete")

    def _reconcile_watch_findings(
        self,
        existing: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        observed_at_ms: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Coalesce repeated localized observations while retaining each evidence link."""
        findings = [dict(item) for item in existing]
        new_findings: list[dict[str, Any]] = []
        observation_updates: list[dict[str, Any]] = []
        for candidate_value in observed:
            candidate = dict(candidate_value)
            match_index = next(
                (
                    index for index, prior in enumerate(findings)
                    if self._same_watch_finding(prior, candidate)
                ),
                None,
            )
            evidence_id = candidate.get("evidence_id")
            frame_id = candidate.get("frame_id")
            observation = {
                "evidence_id": evidence_id,
                "source_binding": candidate.get("source_binding"),
                "frame_id": frame_id,
                "observed_at_ms": int(observed_at_ms),
                "brief_version": candidate.get("brief_version"),
                "brief_sha256": candidate.get("brief_sha256"),
                "model_provenance": dict(candidate.get("model_provenance", {})),
            }
            if match_index is None:
                refs = set(candidate.get("evidence_refs", []))
                if isinstance(evidence_id, str) and evidence_id:
                    refs.add(evidence_id)
                candidate["evidence_refs"] = sorted(refs)
                candidate["observations"] = [observation] if evidence_id else []
                candidate["observation_count"] = len(candidate["observations"])
                candidate["last_observed_evidence_id"] = evidence_id
                candidate["last_observed_frame_id"] = frame_id
                candidate["last_observed_at_ms"] = int(observed_at_ms)
                findings.append(candidate)
                new_findings.append(candidate)
                observation_updates.append({
                    "finding_id": candidate.get("finding_id"),
                    "evidence_id": evidence_id,
                    "frame_id": frame_id,
                    "observation_count": candidate["observation_count"],
                    "new": True,
                })
                continue

            prior = findings[match_index]
            merged = dict(prior)
            refs = set(prior.get("evidence_refs", [])) | set(candidate.get("evidence_refs", []))
            for linked_id in (prior.get("evidence_id"), evidence_id):
                if isinstance(linked_id, str) and linked_id:
                    refs.add(linked_id)
            merged["evidence_refs"] = sorted(refs)
            history = list(prior.get("observations", []))
            if not history and isinstance(prior.get("evidence_id"), str):
                history.append({
                    "evidence_id": prior["evidence_id"],
                    "source_binding": prior.get("source_binding"),
                    "frame_id": prior.get("frame_id"),
                    "brief_version": prior.get("brief_version"),
                    "brief_sha256": prior.get("brief_sha256"),
                    "model_provenance": dict(prior.get("model_provenance", {})),
                })
            if isinstance(evidence_id, str) and evidence_id not in {
                item.get("evidence_id") for item in history if isinstance(item, Mapping)
            }:
                history.append(observation)
            merged["observations"] = history[-MAX_MISSION_EVIDENCE_HISTORY:]
            merged["observation_count"] = max(
                len(history), int(prior.get("observation_count", 0))
            )
            merged["last_observed_evidence_id"] = evidence_id
            merged["last_observed_frame_id"] = frame_id
            merged["last_observed_at_ms"] = int(observed_at_ms)
            # The row remains tied to its first evidence_id. Current geometry
            # and server-authored localization describe the latest observation;
            # every analyzed frame stays available through evidence_refs/history.
            for key in ("items", "item_refs", "localization"):
                if key in candidate:
                    merged[key] = candidate[key]
            if prior.get("review") is None:
                merged["status"] = candidate.get("status", prior.get("status"))
                merged["reason"] = candidate.get("reason", prior.get("reason"))
            findings[match_index] = merged
            observation_updates.append({
                "finding_id": merged.get("finding_id"),
                "evidence_id": evidence_id,
                "frame_id": frame_id,
                "observation_count": merged["observation_count"],
                "new": False,
            })
        return findings, new_findings, observation_updates

    @staticmethod
    def _same_watch_finding(prior: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
        if (
            prior.get("claim_type") != "localized_object"
            or candidate.get("claim_type") != "localized_object"
            or str(prior.get("claim", "")).strip().casefold()
            != str(candidate.get("claim", "")).strip().casefold()
            or prior.get("source_binding") != candidate.get("source_binding")
        ):
            return False
        prior_items = prior.get("items", [])
        candidate_items = candidate.get("items", [])
        for old_item in prior_items:
            old_box = old_item.get("box") if isinstance(old_item, Mapping) else None
            if not isinstance(old_box, (list, tuple)) or len(old_box) != 4:
                continue
            for new_item in candidate_items:
                if not isinstance(new_item, Mapping) or str(old_item.get("label", "")).casefold() != str(new_item.get("label", "")).casefold():
                    continue
                new_box = new_item.get("box")
                if not isinstance(new_box, (list, tuple)) or len(new_box) != 4:
                    continue
                x1 = max(float(old_box[0]), float(new_box[0]))
                y1 = max(float(old_box[1]), float(new_box[1]))
                x2 = min(float(old_box[2]), float(new_box[2]))
                y2 = min(float(old_box[3]), float(new_box[3]))
                intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                old_area = max(0.0, float(old_box[2]) - float(old_box[0])) * max(0.0, float(old_box[3]) - float(old_box[1]))
                new_area = max(0.0, float(new_box[2]) - float(new_box[0])) * max(0.0, float(new_box[3]) - float(new_box[1]))
                union = old_area + new_area - intersection
                if union > 0.0 and intersection / union >= 0.5:
                    return True
        return False

    async def _apply_watch(
        self,
        mission_id: str,
        generation: int,
        binding: SourceBinding | None,
        proposal: WatchProposal,
        frame_received_monotonic: float | None,
    ) -> None:
        if self._closed:
            return
        if binding is None or not isinstance(proposal, WatchProposal):
            return
        targets = tuple(dict.fromkeys(_clean_text(target, limit=MAX_TARGET_CHARS, field="watch target") for target in proposal.targets))
        if not 1 <= len(targets) <= MAX_TARGETS or proposal.task not in {"detect", "segment"}:
            return
        snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
        if self._closed or not self._is_current(snapshot, generation, running=True) or snapshot["mode"] != MODE_WATCH:
            return
        task = snapshot.get("watch_task") or proposal.task
        frame_age = (
            self.clock.monotonic() - frame_received_monotonic
            if isinstance(frame_received_monotonic, (int, float))
            else None
        )
        if frame_age is None or not 0 <= frame_age <= WATCH_MAX_AGE_SECONDS:
            await self._set_waiting(mission_id, generation, "source_unavailable", "The approved source has no fresh frame for Watch configuration.", state="paused")
            return
        expected_configuration_revision = max(
            int(snapshot.get("configuration_revision", 0)),
            self._configuration_revision(),
        )
        request = WatchLeaseRequest(
            mission_id=mission_id,
            mission_revision=int(snapshot["revision"]),
            execution_generation=generation,
            source_binding=binding,
            expected_configuration_revision=expected_configuration_revision,
            targets=targets,
            task=task,
            expires_at_ms=self.clock.now_ms() + WATCH_LEASE_SECONDS * 1000,
        )
        try:
            lease = self.watch_adapter.apply(request)
        except Exception:
            await self._set_waiting(mission_id, generation, "watch_lease_rejected", "Watch target lease was not accepted.", state="paused")
            return
        current = await asyncio.to_thread(self.store.get_mission, mission_id)
        if not self._is_current(current, generation, running=True) or lease.source_binding != binding:
            await self._release_lease(asdict(lease), "stale_watch_apply")
            return
        updated = dict(current)
        updated["watch_lease"] = asdict(lease)
        updated["configuration_revision"] = lease.configuration_revision
        updated["targets"] = list(lease.targets)
        updated["task"] = lease.task
        try:
            committed = await self._change_snapshot(
                mission_id,
                generation,
                updated,
                "watch_lease_applied",
                {"lease_id": lease.lease_id, "targets": list(lease.targets), "task": lease.task},
            )
        except asyncio.CancelledError:
            await asyncio.shield(
                self._release_lease(asdict(lease), "watch_lease_commit_failed")
            )
            raise
        except Exception:
            await self._release_lease(asdict(lease), "watch_lease_commit_failed")
            raise
        if not committed:
            await self._release_lease(asdict(lease), "watch_lease_commit_failed")
            return
        self._schedule_lease_timeout(mission_id, generation, lease)

    async def _change_snapshot(self, mission_id: str, generation: int, updated: Mapping[str, Any], kind: str, data: Mapping[str, Any]) -> dict[str, Any] | None:
        snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
        if not self._is_current(snapshot, generation):
            return None
        now = self.clock.now_ms()
        authoritative_fields = (
            "watch_task",
            "expertise",
            "goal",
            "brief_version",
            "brief_sha256",
            "brief_history",
            "model_provenance",
        )

        def operation(tx):
            current = tx.get_mission(mission_id)
            if (
                not self._is_current(current, generation)
                or int(current["revision"]) != int(snapshot["revision"])
            ):
                return None
            merged = dict(updated)
            for field in authoritative_fields:
                if field in current:
                    merged[field] = current[field]
                else:
                    merged.pop(field, None)
            if kind == "cycle_finished":
                for field in ("watch_lease", "configuration_revision", "targets", "task"):
                    if field in current:
                        merged[field] = current[field]
                    else:
                        merged.pop(field, None)
                for field, identity, current_fields in (
                    ("findings", "finding_id", ("review",)),
                    ("evidence", "evidence_id", ("available", "availability_reason")),
                ):
                    current_rows = {
                        row.get(identity): row
                        for row in current.get(field, ())
                        if isinstance(row, Mapping) and isinstance(row.get(identity), str)
                    }
                    merged_rows = []
                    merged_ids = set()
                    for row in merged.get(field, ()):
                        if not isinstance(row, Mapping):
                            continue
                        value = dict(row)
                        row_id = value.get(identity)
                        latest = current_rows.get(row_id)
                        if latest is not None:
                            for current_field in current_fields:
                                if current_field in latest:
                                    value[current_field] = latest[current_field]
                                else:
                                    value.pop(current_field, None)
                        if isinstance(row_id, str):
                            merged_ids.add(row_id)
                        merged_rows.append(value)
                    merged_rows.extend(
                        dict(row)
                        for row_id, row in current_rows.items()
                        if row_id not in merged_ids
                    )
                    merged[field] = merged_rows
                unavailable_evidence_ids = {
                    row.get("evidence_id")
                    for row in merged.get("evidence", ())
                    if isinstance(row, Mapping)
                    and isinstance(row.get("evidence_id"), str)
                    and row.get("available") is False
                }
                for finding in merged.get("findings", ()):
                    if not isinstance(finding, dict):
                        continue
                    evidence_refs = finding.get("evidence_refs", ())
                    referenced_evidence = set(evidence_refs) if isinstance(evidence_refs, (list, tuple, set)) else set()
                    evidence_id = finding.get("evidence_id")
                    if isinstance(evidence_id, str):
                        referenced_evidence.add(evidence_id)
                    if (
                        finding.get("status") == "supported"
                        and referenced_evidence.intersection(unavailable_evidence_ids)
                    ):
                        finding["status"] = "unresolved"
                        finding["reason"] = "evidence_unavailable"
                    localization = finding.get("localization")
                    if (
                        isinstance(localization, dict)
                        and localization.get("status") == "supported"
                        and localization.get("evidence_id") in unavailable_evidence_ids
                    ):
                        localization["status"] = "unresolved"
                        localization["reason"] = "evidence_unavailable"
            merged["budget"] = current.get("budget", merged.get("budget"))
            merged["last_sequence"] = current.get("last_sequence", merged.get("last_sequence", 0))
            return self._update_with_event(tx, current, merged, kind, data, now)

        try:
            result = await asyncio.to_thread(self.store.transact, operation)
        except RevisionConflict:
            return None
        self._publish_events(result.events)
        return result.value

    async def _pause_after_shutdown(self, mission_id: str) -> Mapping[str, Any] | None:
        def operation(tx):
            snapshot = tx.get_mission(mission_id)
            if not snapshot or snapshot.get("state") not in ACTIVE_STATES:
                return None
            lease = snapshot.get("watch_lease")
            updated = dict(snapshot)
            updated["state"], updated["reason"] = "paused", "runtime_shutdown"
            updated["activity"] = "Runtime stopped; resume explicitly after reconnect."
            updated["execution_generation"] = int(snapshot.get("execution_generation", 0)) + 1
            updated["watch_lease"] = None
            updated["closeup_request"] = None
            self._update_with_event(tx, snapshot, updated, "runtime_shutdown", {}, self.clock.now_ms())
            return lease

        try:
            result = await asyncio.to_thread(self.store.transact, operation)
        except RevisionConflict:
            return None
        self._publish_events(result.events)
        return result.value

    async def _set_waiting(self, mission_id: str, generation: int, reason: str, activity: str, *, state: str = "waiting_evidence") -> None:
        if self._closed:
            return
        snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
        if not self._is_current(snapshot, generation, running=True):
            return
        lease = snapshot.get("watch_lease")
        updated = dict(snapshot)
        updated["state"], updated["reason"] = state, reason
        updated["activity"] = activity
        updated["watch_lease"] = None
        updated["execution_generation"] = generation + 1
        committed = await self._change_snapshot(mission_id, generation, updated, "mission_waiting", {"reason": reason})
        if committed:
            self._interrupt_watch_pacing(mission_id)
        if committed and lease:
            await self._release_lease(lease, reason)

    def _persist_tool_result(
        self,
        mission_id: str,
        generation: int,
        cycle_id: str,
        input_evidence_id: str,
        tool: str,
        result: ToolResult,
        *,
        brief_version: int | None = None,
        brief_sha256: str | None = None,
        model_provenance: Mapping[str, Any] | None = None,
        source_binding: SourceBinding | None = None,
        frame_id: int | None = None,
        input_sha256: str | None = None,
    ) -> tuple[ToolCallRecord, Mapping[str, Any] | None]:
        valid_status = result.status in {"ok", "empty", "unsupported", "failed", "timeout"}
        status = result.status if valid_status else "failed"
        items = tuple(_geometry(item) for item in result.items[:MAX_TOOL_ITEMS])
        text = result.text[:3_000] if isinstance(result.text, str) else ""
        error_code = result.error_code[:128] if isinstance(result.error_code, str) else None
        tool_result_id = uuid.uuid4().hex
        now = self.clock.now_ms()

        def operation(tx):
            snapshot = tx.get_mission(mission_id)
            current = bool(snapshot and int(snapshot.get("execution_generation", -1)) == generation and snapshot.get("state") == "running")
            evidence_ids = []
            evidence_sha256 = {}
            quota_lease = None
            quota_reason = None
            version = int(brief_version if brief_version is not None else (snapshot or {}).get("brief_version", 1))
            digest = brief_sha256 if brief_sha256 is not None else (snapshot or {}).get("brief_sha256")
            provenance = dict(
                model_provenance
                if model_provenance is not None
                else (snapshot or {}).get("model_provenance", {})
            )
            binding = (
                source_binding
                if source_binding is not None
                else self._binding_from_snapshot(snapshot) if snapshot else None
            )
            for artifact in result.artifacts[:4] if current else ():
                if artifact.kind != "crop" or len(artifact.jpeg_bytes) > MAX_DECODED_JPEG_BYTES or artifact.parent_evidence_id != input_evidence_id:
                    continue
                try:
                    room_available = self._make_evidence_room(
                        tx,
                        snapshot,
                        len(artifact.jpeg_bytes),
                        protected_evidence_ids=(input_evidence_id,),
                    )
                    if not room_available:
                        raise QuotaExceeded("no unprotected rolling evidence or byte headroom is available")
                    stored = tx.save_evidence(
                        mission_id,
                        artifact.jpeg_bytes,
                        kind=artifact.kind,
                        origin="generated_crop",
                        parent_evidence_id=artifact.parent_evidence_id,
                        crop_box=artifact.crop_box,
                        input_transform=asdict(artifact.input_transform) if artifact.input_transform else None,
                        created_at_ms=now,
                        brief_version=version,
                        brief_sha256=digest,
                        model_provenance=provenance,
                    )
                except QuotaAccountingIncomplete:
                    quota_reason = "evidence_quota_accounting_unavailable"
                    paused, quota_lease = self._pause_for_evidence_quota(tx, snapshot, now, reason=quota_reason)
                    snapshot = paused
                    current = False
                    break
                except RootQuotaExceeded as exc:
                    self._roll_off_root_only_evidence(
                        tx,
                        snapshot,
                        exc.root_usage_bytes,
                        exc.incoming_bytes,
                        protected_evidence_ids=(input_evidence_id,),
                    )
                    quota_reason = "evidence_quota_full"
                    paused, quota_lease = self._pause_for_evidence_quota(tx, snapshot, now, reason=quota_reason)
                    snapshot = paused
                    current = False
                    break
                except QuotaExceeded:
                    quota_reason = "evidence_quota_full"
                    paused, quota_lease = self._pause_for_evidence_quota(tx, snapshot, now, reason=quota_reason)
                    snapshot = paused
                    current = False
                    break
                evidence_ids.append(stored["evidence_id"])
                evidence_sha256[stored["evidence_id"]] = stored["sha256"]
                if current:
                    snapshot.setdefault("evidence", []).append(stored)
            record = ToolCallRecord(
                tool_result_id,
                tool,
                status,
                input_evidence_id,
                items,
                tuple(evidence_ids),
                text,
                quota_reason or error_code,
                brief_version=version,
                brief_sha256=digest,
                model_provenance=provenance,
                source_binding=asdict(binding) if binding else None,
                frame_id=frame_id,
                input_sha256=input_sha256,
                evidence_sha256=evidence_sha256,
                created_at_ms=now,
            )
            tx.add_tool_record(mission_id, cycle_id, generation, _jsonable(record), current=current, created_at_ms=now)
            if current:
                updated = dict(snapshot)
                updated["activity"] = f"{tool}: {status}"
                updated = tx.update_mission(updated, expected_revision=int(snapshot["revision"]), updated_at_ms=now)
                tx.append_event(mission_id, updated["revision"], "tool_result", {"record": _jsonable(record), "snapshot": self._event_snapshot(updated)}, now)
            else:
                tx.append_event(
                    mission_id,
                    int(snapshot["revision"] if snapshot else 0),
                    "tool_result" if quota_lease else "stale_tool_result",
                    {
                        "record": _jsonable(record),
                        "evidence_quota_full": bool(quota_lease),
                        "snapshot": self._event_snapshot(snapshot) if snapshot else None,
                    },
                    now,
                )
            return record, quota_lease

        result_tx = self.store.transact(operation)
        self._publish_events(result_tx.events)
        return result_tx.value

    def _save_watch_frame(self, mission_id: str, generation: int, frame: SourceFrame) -> tuple[str | None, dict[str, Any]]:
        now = self.clock.now_ms()

        def operation(tx):
            snapshot = tx.get_mission(mission_id)
            if not snapshot or snapshot.get("state") != "running" or int(snapshot.get("execution_generation", -1)) != generation:
                return None, {}
            for existing in snapshot.get("evidence", []):
                if existing.get("available", True) and existing.get("kind") == "frame" and existing.get("source_id") == frame.source_id and existing.get("source_epoch") == frame.source_epoch and existing.get("frame_id") == frame.frame_id:
                    return existing["evidence_id"], existing
            try:
                room_available = self._make_evidence_room(tx, snapshot, len(frame.jpeg_bytes))
                if not room_available:
                    raise QuotaExceeded("no unprotected rolling evidence or byte headroom is available")
                ref = tx.save_evidence(
                    mission_id,
                    frame.jpeg_bytes,
                    kind="frame",
                    origin="watch_frame",
                    source_id=frame.source_id,
                    source_epoch=frame.source_epoch,
                    frame_id=frame.frame_id,
                    capture_time_ms=frame.capture_time_ms,
                    created_at_ms=now,
                    brief_version=int(snapshot.get("brief_version", 1)),
                    brief_sha256=snapshot.get("brief_sha256"),
                    model_provenance=snapshot.get("model_provenance", {}),
                )
            except QuotaAccountingIncomplete:
                reason = "evidence_quota_accounting_unavailable"
                paused, lease = self._pause_for_evidence_quota(tx, snapshot, now, reason=reason)
                return None, {"watch_lease": lease, "snapshot": paused, "release_reason": reason}
            except RootQuotaExceeded as exc:
                self._roll_off_root_only_evidence(
                    tx,
                    snapshot,
                    exc.root_usage_bytes,
                    exc.incoming_bytes,
                )
                reason = "evidence_quota_full"
                paused, lease = self._pause_for_evidence_quota(tx, snapshot, now, reason=reason)
                return None, {"watch_lease": lease, "snapshot": paused, "release_reason": reason}
            except QuotaExceeded:
                reason = "evidence_quota_full"
                paused, lease = self._pause_for_evidence_quota(tx, snapshot, now, reason=reason)
                return None, {"watch_lease": lease, "snapshot": paused, "release_reason": reason}
            updated = dict(snapshot)
            updated.setdefault("evidence", []).append(ref)
            updated["input_evidence_id"] = ref["evidence_id"]
            updated["budget"] = self._budget_for_resume(snapshot.get("budget"), now)
            updated["activity"] = "Analyzing the latest approved source frame."
            updated = tx.update_mission(updated, expected_revision=int(snapshot["revision"]), updated_at_ms=now)
            tx.append_event(mission_id, updated["revision"], "source_frame_accepted", {"evidence": self._public_evidence(ref), "snapshot": self._event_snapshot(updated)}, now)
            return ref["evidence_id"], ref

        result = self.store.transact(operation)
        self._publish_events(result.events)
        return result.value

    async def source_changed(self, source_id: str, source_epoch: str | None, *, available: bool) -> None:
        """Pause and revoke a Watch binding when its accepted source changes."""
        await self._ensure_recovered()
        for snapshot in await asyncio.to_thread(self.store.list_active_missions):
            binding = snapshot.get("source_binding") or {}
            if binding.get("source_id") != source_id or snapshot.get("state") not in ACTIVE_STATES:
                continue
            if available and source_epoch == binding.get("source_epoch"):
                continue
            generation = int(snapshot.get("execution_generation", 0))
            await self._set_waiting(snapshot["mission_id"], generation, "source_disconnected" if not available else "source_epoch_changed", "Source changed; mission paused until explicit resume with the current epoch.", state="paused")

    async def manual_override(self, source_id: str, configuration_revision: int, *, reason: str = "manual_override") -> None:
        """Invalidate adaptive work after a newer operator-owned configuration."""
        await self._ensure_recovered()
        for snapshot in await asyncio.to_thread(self.store.list_active_missions):
            binding = snapshot.get("source_binding") or {}
            if binding.get("source_id") != source_id or snapshot.get("state") not in ACTIVE_STATES:
                continue
            generation = int(snapshot.get("execution_generation", 0))
            updated = dict(snapshot)
            old_lease = updated.get("watch_lease")
            updated["state"], updated["reason"] = "paused", reason[:128]
            updated["activity"] = "Operator configuration changed; adaptive control paused."
            updated["execution_generation"] = generation + 1
            updated["configuration_revision"] = max(int(updated.get("configuration_revision", 0)), int(configuration_revision))
            updated["watch_lease"] = None
            committed = await self._change_snapshot(snapshot["mission_id"], generation, updated, "manual_override", {"configuration_revision": configuration_revision, "reason": reason[:128]})
            if committed:
                self._interrupt_watch_pacing(snapshot["mission_id"])
                if old_lease:
                    await self._release_lease(old_lease, "manual_override")

    async def close(self) -> dict[str, Any]:
        """Invalidate work, wait briefly, and retain planner residency until native drain."""
        if not self._closed:
            self._closed = True
            for mission_id in tuple(self._task_interrupts):
                self._interrupt_watch_pacing(mission_id)
            for mission_id in tuple(self._closeup_timers):
                self._cancel_timer(self._closeup_timers, mission_id)
            for mission_id in tuple(self._lease_timers):
                self._cancel_timer(self._lease_timers, mission_id)
            self._pending_generations.clear()
            for snapshot in await asyncio.to_thread(self.store.list_active_missions):
                lease = await self._pause_after_shutdown(snapshot["mission_id"])
                if lease:
                    await self._release_lease(lease, "runtime_shutdown")
        if self._planner_closed:
            return {"drained": True, "pending_missions": 0, "pending_native_calls": 0}
        pending = {task for task in (*self._tasks.values(), *self._native_calls) if not task.done()}
        if pending:
            _done, pending = await asyncio.wait(pending, timeout=SHUTDOWN_DRAIN_SECONDS)
        if pending:
            if self._close_cleanup is None or self._close_cleanup.done():
                self._close_cleanup = asyncio.create_task(self._close_after_drain(), name="mission-runtime-close-drain")
            return {
                "drained": False,
                "pending_missions": sum(not task.done() for task in self._tasks.values()),
                "pending_native_calls": sum(not task.done() for task in self._native_calls),
            }
        await self._close_planner()
        return {"drained": True, "pending_missions": 0, "pending_native_calls": 0}

    async def _close_after_drain(self) -> None:
        while any(not task.done() for task in (*self._tasks.values(), *self._native_calls)):
            await asyncio.sleep(0.05)
        await self._close_planner()

    async def _close_planner(self) -> None:
        async with self._planner_close_lock:
            if self._planner_closed:
                return
            close = getattr(self.planner, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
            self._planner_closed = True

    async def _ensure_recovered(self) -> None:
        if self._recovered:
            return
        async with self._recovery_lock:
            if self._recovered:
                return
            for snapshot in await asyncio.to_thread(self.store.list_active_missions):
                if snapshot.get("state") not in ACTIVE_STATES:
                    continue
                generation = int(snapshot.get("execution_generation", 0))
                lease = snapshot.get("watch_lease")
                updated = dict(snapshot)
                updated["state"], updated["reason"] = "paused", "recovery_required"
                updated["activity"] = "Previous runtime ended; review state and resume explicitly."
                updated["execution_generation"] = generation + 1
                updated["watch_lease"] = None
                updated["closeup_request"] = None
                committed = await self._change_snapshot(snapshot["mission_id"], generation, updated, "recovery_required", {})
                if committed and lease:
                    await self._release_lease(lease, "recovery_required")
            self._recovered = True

    def _is_current(self, snapshot: Mapping[str, Any] | None, generation: int, *, running: bool = False) -> bool:
        return bool(snapshot and int(snapshot.get("execution_generation", -1)) == generation and (not running or snapshot.get("state") == "running"))

    @staticmethod
    def _binding_from_snapshot(snapshot: Mapping[str, Any]) -> SourceBinding | None:
        value = snapshot.get("source_binding")
        if not value:
            return None
        try:
            return _source_binding(value)
        except _CommandError:
            return None

    def _configuration_revision(self) -> int:
        revision = self.watch_adapter.current_configuration_revision()
        if not _integer(revision) or revision < 0:
            raise _CommandError("control_state_unavailable", "bridge configuration revision is invalid")
        return revision

    def _frame_matches(self, frame: SourceFrame | None, binding: SourceBinding, *, max_age: float) -> bool:
        if frame is None or not isinstance(frame, SourceFrame):
            return False
        age = self.clock.monotonic() - frame.received_monotonic
        return bool(
            frame.source_id == binding.source_id
            and frame.source_epoch == binding.source_epoch
            and 0 <= age <= max_age
            and isinstance(frame.jpeg_bytes, bytes)
            and 0 < len(frame.jpeg_bytes) <= MAX_DECODED_JPEG_BYTES
            and _integer(frame.frame_id)
        )

    def _evidence_metadata(self, evidence_id: str) -> dict[str, Any]:
        return self.store.read_evidence(evidence_id)["metadata"]

    def _evidence_owner(self, evidence_id: str) -> str | None:
        return self.store.evidence_owner(evidence_id)

    def _verified_evidence(self, evidence_id: str, mission_id: str, *, tx=None) -> dict[str, Any]:
        owner = self._evidence_owner(evidence_id)
        if owner != mission_id:
            raise EvidenceUnavailable("evidence is missing or belongs to another mission")
        try:
            return self.store.read_evidence(evidence_id)
        except EvidenceUnavailable as exc:
            if tx is not None:
                self._mark_evidence_unavailable_in_tx(
                    tx, mission_id, evidence_id, exc.availability_reason
                )
            else:
                self._mark_evidence_unavailable(mission_id, evidence_id, exc.availability_reason)
            raise EvidenceUnavailable(
                "evidence is missing or changed", availability_reason=exc.availability_reason
            ) from exc
        except KeyError as exc:
            if tx is not None:
                self._mark_evidence_unavailable_in_tx(tx, mission_id, evidence_id, "missing")
            else:
                self._mark_evidence_unavailable(mission_id, evidence_id, "missing")
            raise EvidenceUnavailable("evidence is missing or changed", availability_reason="missing") from exc

    def _verify_snapshot_evidence(
        self,
        mission_id: str,
        snapshot: Mapping[str, Any],
        *,
        tx=None,
        verify_content: bool = False,
    ) -> dict[str, Any]:
        unavailable = []
        for ref in snapshot.get("evidence", []):
            try:
                if verify_content:
                    self.store.read_evidence(ref["evidence_id"])
                else:
                    self.store.check_evidence_file(ref["evidence_id"])
            except EvidenceUnavailable as exc:
                unavailable.append((ref["evidence_id"], exc.availability_reason))
            except KeyError:
                unavailable.append((ref["evidence_id"], "missing"))
        if tx is not None:
            current = dict(snapshot)
            for evidence_id, reason in unavailable:
                current = self._mark_evidence_unavailable_in_tx(tx, mission_id, evidence_id, reason) or current
            return tx.get_mission(mission_id) or current
        for evidence_id, reason in unavailable:
            self._mark_evidence_unavailable(mission_id, evidence_id, reason)
        return self.store.get_mission(mission_id) or dict(snapshot)

    def _mark_evidence_unavailable_in_tx(
        self, tx, mission_id: str, evidence_id: str, reason: str = "missing"
    ) -> dict[str, Any] | None:
        snapshot = tx.get_mission(mission_id)
        if not snapshot:
            return None
        updated = dict(snapshot)
        refs = [dict(item) for item in updated.get("evidence", [])]
        changed = tx.set_evidence_unavailable(mission_id, evidence_id, reason)
        for ref in refs:
            if ref.get("evidence_id") == evidence_id and (
                ref.get("available") is not False or ref.get("availability_reason") != reason
            ):
                ref["available"] = False
                ref["availability_reason"] = reason
                changed = True
        for finding in updated.get("findings", []):
            if evidence_id in finding.get("evidence_refs", []) or finding.get("evidence_id") == evidence_id:
                if finding.get("status") == "supported":
                    finding["status"] = "unresolved"
                    finding["reason"] = "evidence_unavailable"
                    changed = True
                localization = finding.get("localization")
                if isinstance(localization, dict) and localization.get("evidence_id") == evidence_id and localization.get("status") == "supported":
                    localization["status"] = "unresolved"
                    localization["reason"] = "evidence_unavailable"
                    changed = True
        if not changed:
            return snapshot
        updated["evidence"] = refs
        now = self.clock.now_ms()
        return self._update_with_event(tx, snapshot, updated, "evidence_unavailable", {"evidence_id": evidence_id}, now)

    def _mark_evidence_unavailable(
        self, mission_id: str, evidence_id: str, reason: str = "missing"
    ) -> None:
        try:
            result = self.store.transact(
                lambda tx: self._mark_evidence_unavailable_in_tx(tx, mission_id, evidence_id, reason)
            )
        except RevisionConflict:
            return
        self._publish_events(result.events)

    def _public_evidence(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return {key: val for key, val in value.items() if key not in {"path", "jpeg_bytes"}}

    def _protected_evidence_ids(
        self, snapshot: Mapping[str, Any], extra: Collection[str] = ()
    ) -> set[str]:
        protected = {value for value in extra if isinstance(value, str) and value}
        for value in (
            snapshot.get("input_evidence_id"),
            (snapshot.get("closeup_request") or {}).get("evidence_id")
            if isinstance(snapshot.get("closeup_request"), Mapping)
            else None,
        ):
            if isinstance(value, str) and value:
                protected.add(value)
        for finding in snapshot.get("findings", ()):
            if not isinstance(finding, Mapping):
                continue
            for value in (
                finding.get("evidence_id"),
                finding.get("last_observed_evidence_id"),
            ):
                if isinstance(value, str) and value:
                    protected.add(value)
            protected.update(
                value for value in finding.get("evidence_refs", ())
                if isinstance(value, str) and value
            )
            for observation in finding.get("observations", ()):
                if isinstance(observation, Mapping):
                    value = observation.get("evidence_id")
                    if isinstance(value, str) and value:
                        protected.add(value)
        return protected

    @staticmethod
    def _apply_evidence_tombstone(snapshot: dict[str, Any], tombstone: Mapping[str, Any]) -> None:
        evidence_id = tombstone.get("evidence_id")
        refs = [dict(item) for item in snapshot.get("evidence", ()) if isinstance(item, Mapping)]
        matching = False
        for ref in refs:
            if ref.get("evidence_id") == evidence_id:
                ref.update(available=False, availability_reason="rolled_off")
                matching = True
        if not matching:
            refs.append(dict(tombstone))
        snapshot["evidence"] = refs
        cycles = [dict(item) for item in snapshot.get("cycle_history", ()) if isinstance(item, Mapping)]
        for cycle in cycles:
            cycle_refs = []
            for value in cycle.get("evidence_refs", ()):
                ref = dict(value) if isinstance(value, Mapping) else {"evidence_id": value}
                if ref.get("evidence_id") == evidence_id:
                    ref.update(available=False, availability_reason="rolled_off")
                cycle_refs.append(ref)
            cycle["evidence_refs"] = cycle_refs
        snapshot["cycle_history"] = cycles

    def _make_evidence_room(
        self,
        tx,
        snapshot: dict[str, Any],
        incoming_bytes: int,
        *,
        protected_evidence_ids: Collection[str] = (),
    ) -> bool:
        """Roll only oldest unprotected frame/crop media before a new save."""
        if len(snapshot.get("evidence", ())) >= MAX_MISSION_EVIDENCE_HISTORY:
            return False
        if incoming_bytes <= 0 or incoming_bytes > int(self.store.quota_bytes):
            return False
        protected = self._protected_evidence_ids(snapshot, protected_evidence_ids)
        mission_count, _mission_bytes = tx.evidence_usage(snapshot["mission_id"])
        total_bytes = tx.total_evidence_usage()[1]
        while (
            mission_count >= MAX_MISSION_EVIDENCE
            or total_bytes + incoming_bytes > int(self.store.quota_bytes)
        ):
            candidates = [
                ref for ref in tx.list_evidence(snapshot["mission_id"])
                if ref.get("evidence_id")
                and tx._is_evidence_rolloff_eligible(str(ref["evidence_id"]))
                and ref.get("evidence_id") not in protected
                and not tx.evidence_is_exported(str(ref.get("evidence_id", "")))
                and (
                    any(
                        current.get("evidence_id") == ref.get("evidence_id")
                        for current in snapshot.get("evidence", ())
                    )
                    or len(snapshot.get("evidence", ())) < MAX_MISSION_EVIDENCE_HISTORY - 1
                )
            ]
            if not candidates:
                return False
            candidate = candidates[0]
            tombstone = tx.retire_evidence(str(candidate["evidence_id"]))
            if tombstone is None:
                continue
            self._apply_evidence_tombstone(snapshot, tombstone)
            mission_count, _mission_bytes = tx.evidence_usage(snapshot["mission_id"])
            total_bytes = tx.total_evidence_usage()[1]
        return True

    def _roll_off_root_only_evidence(
        self,
        tx,
        snapshot: dict[str, Any],
        root_usage_bytes: int,
        incoming_bytes: int,
        *,
        protected_evidence_ids: Collection[str] = (),
    ) -> None:
        """Tombstone oldest safe crops and automatic Watch frames after root refusal."""
        bytes_to_clear = root_usage_bytes + incoming_bytes - int(self.store.root_quota_bytes)
        if bytes_to_clear <= 0:
            return
        protected = self._protected_evidence_ids(snapshot, protected_evidence_ids)
        for candidate in tx.list_evidence(snapshot["mission_id"]):
            evidence_id = str(candidate.get("evidence_id", ""))
            if (
                not evidence_id
                or not tx._is_evidence_rolloff_eligible(evidence_id)
                or evidence_id in protected
                or tx.evidence_is_exported(evidence_id)
                or (
                    not any(ref.get("evidence_id") == evidence_id for ref in snapshot.get("evidence", ()))
                    and len(snapshot.get("evidence", ())) >= MAX_MISSION_EVIDENCE_HISTORY - 1
                )
            ):
                continue
            size_bytes = int(candidate.get("bytes", 0))
            if size_bytes <= 0:
                continue
            tombstone = tx.retire_evidence(evidence_id)
            if tombstone is None:
                continue
            self._apply_evidence_tombstone(snapshot, tombstone)
            # Stored sizes bound phase-one cleanup only. The next save performs a
            # fresh root scan; these pending unlinks are never admission credit.
            bytes_to_clear -= size_bytes
            if bytes_to_clear <= 0:
                break

    def _pause_for_evidence_quota(
        self,
        tx,
        snapshot: Mapping[str, Any],
        now: int,
        *,
        reason: str = "evidence_quota_full",
        preserve_closeup_wait: bool = False,
    ):
        updated = dict(snapshot)
        lease = updated.get("watch_lease")
        updated["state"], updated["reason"] = ("waiting_evidence" if preserve_closeup_wait else "paused"), reason
        if reason == "evidence_quota_accounting_unavailable":
            updated["activity"] = (
                "Evidence storage could not be accounted safely; resolve the storage issue, then retry the requested close-up."
                if preserve_closeup_wait else
                "Evidence storage could not be accounted safely; resolve the storage issue, then resume explicitly."
            )
        else:
            updated["activity"] = (
                "Evidence storage is at its configured limit; free space or remove eligible evidence, then retry the requested close-up."
                if preserve_closeup_wait else
                "Evidence storage is at its configured limit; free space or remove eligible evidence, then resume explicitly."
            )
        updated["execution_generation"] = int(snapshot.get("execution_generation", 0)) + 1
        updated["cycle_id"] = None
        updated["watch_lease"] = None
        saved = self._update_with_event(
            tx,
            snapshot,
            updated,
            "mission_waiting",
            {"reason": reason},
            now,
        )
        return saved, lease

    async def _release_lease(
        self,
        value: Mapping[str, Any],
        reason: str,
        *,
        cancel_timer: bool = True,
    ) -> WatchLeaseRelease | None:
        mission_id = value.get("mission_id") if isinstance(value, Mapping) else None
        if cancel_timer and isinstance(mission_id, str):
            self._cancel_timer(self._lease_timers, mission_id)
        try:
            lease = value if isinstance(value, WatchLease) else WatchLease(
                lease_id=str(value["lease_id"]),
                mission_id=str(value["mission_id"]),
                source_binding=_source_binding(value["source_binding"]),
                configuration_revision=int(value["configuration_revision"]),
                targets=tuple(value["targets"]),
                task=value["task"],
                expires_at_ms=int(value["expires_at_ms"]),
            )
            result = self.watch_adapter.release(lease, reason)
            return result if isinstance(result, WatchLeaseRelease) else None
        except Exception:
            return None

    @staticmethod
    def _cancel_timer(timers: dict[str, asyncio.Task[None]], key: str) -> None:
        task = timers.pop(key, None)
        if task is not None and not task.done():
            task.cancel()

    def _schedule_closeup_timeout(self, mission_id: str, request_id: str, expires_at_ms: int) -> None:
        self._cancel_timer(self._closeup_timers, mission_id)
        task = asyncio.create_task(self._expire_closeup(mission_id, request_id, expires_at_ms))
        self._closeup_timers[mission_id] = task
        task.add_done_callback(lambda done, key=mission_id: self._closeup_timers.pop(key, None) if self._closeup_timers.get(key) is done else None)

    async def _expire_closeup(self, mission_id: str, request_id: str, expires_at_ms: int) -> None:
        try:
            await asyncio.sleep(max(0.0, (expires_at_ms - self.clock.now_ms()) / 1000))
            snapshot = await asyncio.to_thread(self.store.get_mission, mission_id)
            closeup = snapshot.get("closeup_request") if snapshot else None
            if not snapshot or snapshot.get("state") != "waiting_evidence" or not closeup or closeup.get("request_id") != request_id:
                return
            updated = dict(snapshot)
            updated["state"], updated["reason"] = "paused", "closeup_expired"
            updated["activity"] = "The five-minute close-up window expired; the finding remains unresolved. Resume explicitly to continue."
            updated["execution_generation"] = int(snapshot.get("execution_generation", 0)) + 1
            updated["closeup_request"] = None
            result = await asyncio.to_thread(
                self.store.transact,
                lambda tx: self._update_with_event(tx, snapshot, updated, "closeup_expired", {"reason": "closeup_expired"}, self.clock.now_ms()),
            )
            self._publish_events(result.events)
        except asyncio.CancelledError:
            raise
        except Exception:
            return

    def _schedule_lease_timeout(self, mission_id: str, generation: int, lease: WatchLease) -> None:
        self._cancel_timer(self._lease_timers, mission_id)
        task = asyncio.create_task(self._expire_lease(mission_id, generation, lease))
        self._lease_timers[mission_id] = task
        task.add_done_callback(lambda done, key=mission_id: self._lease_timers.pop(key, None) if self._lease_timers.get(key) is done else None)

    async def _expire_lease(self, mission_id: str, generation: int, lease: WatchLease) -> None:
        try:
            await asyncio.sleep(max(0.0, (lease.expires_at_ms - self.clock.now_ms()) / 1000))
            now = self.clock.now_ms()

            def operation(tx):
                snapshot = tx.get_mission(mission_id)
                current_lease = snapshot.get("watch_lease") if snapshot else None
                if (
                    not self._is_current(snapshot, generation, running=True)
                    or not current_lease
                    or current_lease.get("lease_id") != lease.lease_id
                    or int(current_lease.get("expires_at_ms", 0)) != lease.expires_at_ms
                ):
                    return None
                updated = dict(snapshot)
                updated["state"], updated["reason"] = "paused", "watch_lease_expired"
                updated["activity"] = "The temporary Watch target lease expired; review and resume explicitly."
                updated["execution_generation"] = generation + 1
                updated["watch_lease"] = None
                return self._update_with_event(tx, snapshot, updated, "mission_waiting", {"reason": "watch_lease_expired"}, now)

            result = await asyncio.to_thread(self.store.transact, operation)
            self._publish_events(result.events)
            if result.value:
                self._lease_timers.pop(mission_id, None)
                await self._release_lease(asdict(lease), "watch_lease_expired")
        except asyncio.CancelledError:
            raise
        except Exception:
            return
