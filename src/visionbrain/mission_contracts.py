"""Typed, MLX-free contracts for the adaptive mission runtime.

This module is the Python side of the additive ``mission.v1`` profile. It must
remain importable on bridge/test hosts that do not have MLX or model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence, TypeAlias

MISSION_SCHEMA_VERSION = 1
PROFILE_ID = "visual_inspection"
PROFILE_VERSION = 1

MODE_INSPECT = "inspect"
MODE_WATCH = "watch"
MissionMode: TypeAlias = Literal["inspect", "watch"]

MISSION_STATES = frozenset(
    {
        "created",
        "running",
        "waiting_evidence",
        "waiting_approval",
        "paused",
        "completed",
        "failed",
        "cancelled",
    }
)
TOOL_NAMES = frozenset(
    {
        "detect_objects",
        "segment_objects",
        "inspect_crop",
        "read_text",
        "request_closeup",
        "finish",
    }
)
PERCEPTION_TOOL_NAMES = frozenset(
    {"detect_objects", "segment_objects", "inspect_crop", "read_text"}
)

SCOPE_MISSION_READ = "mission:read"
SCOPE_MISSION_CONTROL = "mission:control"
SCOPE_MISSION_EVIDENCE = "mission:evidence"
SCOPE_MISSION_REVIEW = "mission:review"
MISSION_SCOPES = frozenset(
    {
        SCOPE_MISSION_READ,
        SCOPE_MISSION_CONTROL,
        SCOPE_MISSION_EVIDENCE,
        SCOPE_MISSION_REVIEW,
    }
)

MAX_EXPERTISE_CHARS = 1_000
MAX_GOAL_CHARS = 1_000
MAX_DECODED_JPEG_BYTES = 2 * 1024 * 1024
MAX_OUTBOUND_MESSAGE_BYTES = 256 * 1024
MAX_TARGETS = 8
MAX_TARGET_CHARS = 64
MAX_TOOL_QUESTION_CHARS = 500
MAX_FINDINGS = 8
MAX_TOOL_ITEMS = 32
MAX_POLYGON_POINTS = 64
MAX_TOOL_CALLS_PER_CYCLE = 10
MAX_GENERATIONS_PER_CYCLE = 6
MAX_ACTIVE_SECONDS = 180
MAX_HUMAN_WAIT_SECONDS = 300
MAX_GENERATIONS_PER_WINDOW = 60
MAX_TOOL_CALLS_PER_WINDOW = 100
USAGE_WINDOW_MS = 300_000
MAX_MISSION_EVIDENCE = 64
MAX_MISSION_EVIDENCE_HISTORY = 384
MAX_MISSION_FINDINGS = 16
MAX_FINDING_ITEMS = 4
MAX_BRIEF_VERSIONS = 64
MAX_BRIEF_FULL_TEXT_VERSIONS = 8
MAX_MISSION_CYCLES = 384
WATCH_LEASE_SECONDS = 45
WATCH_MIN_INTERVAL_SECONDS = 10
WATCH_HEARTBEAT_SECONDS = 15
WATCH_MAX_AGE_SECONDS = 15


@dataclass(frozen=True)
class SourceBinding:
    """Server-issued producer identity selected by a mission."""

    source_id: str
    source_epoch: str


@dataclass(frozen=True)
class SourceFrame:
    """Immutable, server-accepted frame supplied to one Watch cycle."""

    jpeg_bytes: bytes
    source_id: str
    source_epoch: str
    frame_id: int
    received_monotonic: float
    capture_time_ms: int | None = None


@dataclass(frozen=True)
class InputTransform:
    """Optional mapping from the normalized input JPEG to its source image."""

    origin_width: int | None = None
    origin_height: int | None = None
    rotation_degrees: int = 0
    resized: bool = False
    scale_x: float | None = None
    scale_y: float | None = None
    flipped: bool = False
    width: int | None = None
    height: int | None = None


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    sha256: str
    width: int
    height: int
    kind: str = "original"
    parent_evidence_id: str | None = None
    crop_box: tuple[float, float, float, float] | None = None
    input_transform: InputTransform | None = None


@dataclass(frozen=True)
class GeometryItem:
    """Grounded normalized geometry emitted by a perception tool."""

    item_id: str
    label: str
    score: float
    box: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...] | None = None
    source: str = "visionbrain"


@dataclass(frozen=True)
class EvidenceArtifact:
    """Image bytes produced by a tool; the store assigns its opaque ID/hash."""

    jpeg_bytes: bytes
    kind: str
    parent_evidence_id: str
    crop_box: tuple[float, float, float, float] | None = None
    input_transform: InputTransform | None = None


@dataclass(frozen=True)
class ToolCallRecord:
    """Bounded result summary exposed to later planner turns."""

    tool_result_id: str
    tool: str
    status: Literal["ok", "empty", "unsupported", "failed", "timeout"]
    input_evidence_id: str
    items: tuple[GeometryItem, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    text: str = ""
    error_code: str | None = None
    brief_version: int = 1
    brief_sha256: str | None = None
    model_provenance: Mapping[str, Any] = field(default_factory=dict)
    unavailable_evidence_ids: tuple[str, ...] = ()
    source_binding: Mapping[str, Any] | None = None
    frame_id: int | None = None
    input_sha256: str | None = None
    evidence_sha256: Mapping[str, str] = field(default_factory=dict)
    created_at_ms: int | None = None


@dataclass(frozen=True)
class FindingProposal:
    """Untrusted claim proposal; runtime assigns support state after validation."""

    claim: str
    claim_type: Literal["localized_object", "text_read", "visual_hypothesis"]
    evidence_refs: tuple[str, ...]
    item_refs: tuple[tuple[str, str], ...] = ()
    text_refs: tuple[str, ...] = ()
    relevance: str = ""


@dataclass(frozen=True)
class WatchProposal:
    targets: tuple[str, ...]
    task: Literal["detect", "segment"]


@dataclass(frozen=True)
class Decision:
    """One typed planner action; arbitrary prose is never executable."""

    schema_version: int
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    findings: tuple[FindingProposal, ...] = ()
    watch: WatchProposal | None = None


@dataclass(frozen=True)
class PlannerContext:
    """One bounded planner turn, including the actual cycle image bytes."""

    mission_id: str
    revision: int
    cycle_id: str
    execution_generation: int
    profile_id: str
    profile_version: int
    mode: MissionMode
    expertise: str
    goal: str
    reasoning_model: str
    input_evidence_id: str
    input_sha256: str
    image_jpeg: bytes
    image_width: int
    image_height: int
    source_binding: SourceBinding | None
    frame_id: int | None
    input_transform: InputTransform | None
    allowed_tools: tuple[Mapping[str, Any], ...]
    tool_results: tuple[ToolCallRecord, ...]
    findings: tuple[Mapping[str, Any], ...]
    generations_remaining: int
    tool_calls_remaining: int
    repair_feedback: str | None = None
    brief_version: int = 1
    brief_sha256: str | None = None
    model_provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolRequest:
    """Validated tool arguments bound to the active cycle's immutable input."""

    tool: str
    arguments: Mapping[str, Any]
    mission_id: str
    cycle_id: str
    execution_generation: int
    reasoning_model: str
    input_evidence_id: str
    input_sha256: str
    frame_id: int | None
    source_binding: SourceBinding | None
    brief_version: int = 1
    brief_sha256: str | None = None
    model_provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolContext:
    """Native-tool context pinned to the same image as ``ToolRequest``."""

    image_jpeg: bytes
    image_width: int
    image_height: int
    grounded_items: Mapping[str, GeometryItem]
    prior_results: tuple[ToolCallRecord, ...]
    input_transform: InputTransform | None = None


@dataclass(frozen=True)
class ToolResult:
    """Explicit perception outcome; empty, failure, and unsupported differ."""

    status: Literal["ok", "empty", "unsupported", "failed", "timeout"]
    items: tuple[GeometryItem, ...] = ()
    artifacts: tuple[EvidenceArtifact, ...] = ()
    text: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None


@dataclass(frozen=True)
class WatchLeaseRequest:
    """Validated temporary target configuration proposed by the planner."""

    mission_id: str
    mission_revision: int
    execution_generation: int
    source_binding: SourceBinding
    expected_configuration_revision: int
    targets: tuple[str, ...]
    task: Literal["detect", "segment"]
    expires_at_ms: int


@dataclass(frozen=True)
class WatchLease:
    """Acknowledged lease returned by the bridge control arbiter."""

    lease_id: str
    mission_id: str
    source_binding: SourceBinding
    configuration_revision: int
    targets: tuple[str, ...]
    task: Literal["detect", "segment"]
    expires_at_ms: int


@dataclass(frozen=True)
class WatchLeaseRelease:
    """Result of a compare-and-release; false means newer control owns config."""

    released: bool
    configuration_revision: int | None = None
    reason: str | None = None


@dataclass(frozen=True)
class MissionEvent:
    mission_id: str
    revision: int
    sequence: int
    kind: str
    data: Mapping[str, Any]
    timestamp_ms: int


@dataclass(frozen=True)
class Principal:
    """Authenticated server principal; actor IDs never come from command args."""

    principal_id: str
    source_id: str | None = None
    source_epoch: str | None = None
    frame_id: int | None = None
    received_monotonic: float | None = None
    capture_time_ms: int | None = None
    evidence_kind: str = "imported"


class Planner(Protocol):
    """Structured planner adapter; availability must not trigger a download."""

    def available(self, model_key: str) -> bool:
        """Return whether this installed model can currently serve a plan."""

    def plan(self, context: PlannerContext) -> Decision:
        """Return one typed decision grounded in the context image."""


class MissionTool(Protocol):
    """One native perception operation; runtime validates before calling it."""

    def execute(self, request: ToolRequest, context: ToolContext) -> ToolResult:
        """Run one bounded tool against the pinned image and cycle references."""


class SourceProvider(Protocol):
    def __call__(self, binding: SourceBinding) -> SourceFrame | None:
        """Return the current accepted frame only for the approved source epoch."""


class WatchAdapter(Protocol):
    def current_configuration_revision(self) -> int:
        """Return the arbiter's current revision for the active source."""

    def apply(self, request: WatchLeaseRequest) -> WatchLease:
        """Apply one temporary target lease under the bridge arbiter."""

    def release(self, lease: WatchLease, reason: str) -> WatchLeaseRelease:
        """Release only if the same lease still owns the applied revision."""


class EventSink(Protocol):
    def __call__(self, event: MissionEvent) -> None:
        """Publish an already persisted mission event to negotiated clients."""


class Clock(Protocol):
    def now_ms(self) -> int:
        """Return wall-clock milliseconds for durable event timestamps."""

    def monotonic(self) -> float:
        """Return monotonic seconds for deadlines and same-host durations."""


SourceProviderFn: TypeAlias = Callable[[SourceBinding], SourceFrame | None]
ToolMap: TypeAlias = Mapping[str, MissionTool]
QualifiedModels: TypeAlias = Mapping[str, Sequence[str]]
