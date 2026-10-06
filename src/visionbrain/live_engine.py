"""Local live engine — a WebSocket endpoint that streams SAM 3.1 detections.

Purpose
-------
The live tab normally connects as a WebSocket *client* to an external field
hub. This module lets the VisionBrain app itself play that server role: it
runs SAM 3.1 detection over an uploaded video file, a local webcam, or a
network stream (rtsp/rtspS/http(s) URL, credentials redacted everywhere a
URL is displayed) and
streams the SAME wire protocol, so the existing browser client works
unchanged. Frames are sent CLEAN (no overlay) — the client draws boxes from
the detection items itself. On top of the raw stream it adds smart capture:
named zones, deterministic triggers (line-cross / direction / dwell), a VLM
event watch, and server-side clip capture.

Wire format (binary frames, byte-identical to the field hub)
------------------------------------------------------------
    struct ">III"   -> (frame_id u32, timestamp_ms u32, jpeg_len u32)
    jpeg bytes      -> jpeg_len bytes
    struct ">I"     -> telem_len u32
    telemetry JSON  -> telem_len bytes (utf-8)

Outbound JSON text messages:

    {"type": "status", "note": str}
    {"type": "detections", "items": [{"box": [x1, y1, x2, y2] (normalized
        0-1), "label": str, "score": float, "track_id": int,
        "color_id": int, "direction": str, "track_state"?: str,
        "polygon"?: [[x, y], ...] normalized mask outline (present when SAM
        produced a mask for the object — clients paint it as a filled shape
        and fall back to the box when absent)}]}
    {"type": "engine_stopped"}
    {"type": "event", "event": {"kind": "line_cross" | "direction" | "dwell"
        | "watch" | "zone_enter" | "zone_exit", "zone": str, "direction": str,
        "track_id": int | null, "ts": float, "frame_id": int, "detail": str}}
    {"type": "capture", "clip": {"name": str, "url": "/api/clips/<name>",
        "kind": str}}
    {"type": "error", "error": str}     -> ask/report path ONLY: the request
        was rejected (no engine running, another ask/report already running,
        or no usable evidence) or backend VLM inference failed. Everything
        else keeps its "status" notes — no client should parse status text
        to detect an ask/report failure.

Clips are written by the worker to the configured clips directory (see
``configure()``) and served by ``web_app`` at ``GET /api/clips/{name}``.

Control protocol (inbound JSON text, one action per message)
------------------------------------------------------------
The control key is ``"type"`` (hub-protocol style); ``"action"`` is accepted
as an alias for programmatic callers.

    {"type": "start", "source": "file", "file_id": str,
     "prompts": [str, ...]}              -> run on an uploaded file
    {"type": "start", "source": "webcam", "camera": int >= 0,
     "prompts": [str, ...]}              -> run on a local camera
    {"type": "start", "source": "url", "url": str,
     "prompts": [str, ...]}              -> run on a network stream
        (rtsp://, rtsps://, http://, https:// — see ``validate_stream_url``;
        URLs are always redacted via ``redact_url`` before display)
    {"type": "set_prompts", "prompts": [str, ...]} -> swap prompts live.
        An EMPTY list is valid and means "detection off" (the hub-protocol
        pause semantics: prompts are preserved client-side and restored on
        resume); while off the worker skips the model entirely and emits
        empty detection sets.
    {"type": "set_threshold", "threshold": float 0-1}
        -> live re-apply of the detection threshold (no restart).
    {"type": "set_stream", "jpeg_quality"?: int 30-95,
     "send_width"?: int 256-3840}
        -> live retune of the outbound JPEG encode (quality + downscale
            width). Lower quality/width cuts encode CPU and wire size with
            no effect on detection quality — the same STREAM knobs the
            Android clients expose.
    {"type": "set_task", "task": "segment"|"detect"}
        -> live MASK switch: "segment" emits per-object polygons (slower,
            larger payloads); "detect" skips polygon tracing for fast boxes.
    {"type": "set_engine", "sam"?: bool, "falcon"?: bool, "lfm"?: bool,
     "lfm_model"?: "lfm"|"lfm3b"}
        -> hub-protocol engine chips; the local engine runs SAM only and
            answers with an honest status note (never a silent no-op).
    {"type": "set_vlm", "model": "gemma"|"lfm"|"lfm3b"}
        -> select the ask/report VLM (cheap; swap applies on next ask).
    {"type": "ask", "question": str <= 500}
        -> ask the selected VLM about the CURRENT OBSERVATION: at dispatch
            the handler snapshots (frame, detection records, active prompts)
            as ONE consistent unit and the background thread answers from
            that snapshot alone. Replies as an ``ask_ack`` followed by an
            ``answer`` message — or ``error`` on rejection/failure.
    {"type": "report", "summary"?: str, "report_type"?: str}
        -> write a grounded field report. The counts line is derived
            SERVER-SIDE from the evidence snapshot's detection records; the
            browser-supplied ``summary`` is accepted and validated for wire
            compatibility but is never forwarded to the model. Replies as a
            ``report_result`` message — or ``error`` on rejection/failure.
    {"type": "set_zones", "zones": [...]}  -> REPLACE the whole zone set.
        Each zone: {"kind": "line", "name"?: str, "a": [x, y], "b": [x, y]}
        or {"kind": "rect", "name"?: str, "x1", "y1", "x2", "y2"} — all
        coordinates normalized 0-1, max 8 zones. Accepted before a worker
        exists (held pending, applied on start) and while running (picked up
        under the state lock on the next detect frame).
    {"type": "add_prompt_box", "box": [x1, y1, x2, y2], "label"?: str}
        -> add a box-prompted target: a persistent region of interest the
            engine re-detects every detect frame alongside the text prompts,
            relabeled "label" (<= 40 chars) or the engine-side default
            "target N" (N = running per-engine sequence). The box is
            normalized 0-1 xyxy with x1 < x2 and y1 < y2 (see
            ``validate_box``). Max 8 concurrent targets. Accepted before a
            worker exists (held pending, applied on start) and while running
            (picked up on the next detect frame).
    {"type": "remove_targets"}           -> clear all box-prompted targets.
    {"type": "set_triggers", "line_cross"?: bool,
     "direction"?: "none"|"any"|"north"|"northeast"|"east"|"southeast"|
     "south"|"southwest"|"west"|"northwest", "dwell_s"?: number >= 0,
     "clip"?: bool, "pre_s"?: number >= 0, "post_s"?: number >= 0}
        -> arm deterministic triggers; all keys optional, merged over the
        current config (defaults: line_cross False, direction "none",
        dwell_s 0 = off, clip True, pre_s 6, post_s 4).
    {"type": "set_watch", "enabled"?: bool, "condition"?: str,
     "interval_s"?: number 1-30, "model"?: "lfm"|"lfm3b"}
        -> start/stop the VLM event watch thread.
    {"type": "stop"}                     -> exit after the current frame
    {"type": "shutdown"}                 -> stop + free the SAM 3.1 model

Optional "start" tuning keys: "threshold" (0-1, default 0.15),
"detect_every" (int >= 1, default 6), "resolution" (int, default 1008),
"backbone_every" (DEPRECATED — accepted and validated for compatibility
with older clients, then ignored: image features are recomputed on EVERY
detection pass and a status note says so), "jpeg_quality" (30-95,
default 70), "send_width" (pixels, default 1280),
"task" ("segment"|"detect", default "segment").
Anything that fails validation returns ("unknown", {}) and earns a status
note — never an exception.

Design notes
------------
* ONE engine worker per process: a module-level handle guarded by
  ``_engine_lock``. MULTI-VIEWER: every connected socket gets a sink and the
  engine broadcasts to all of them, so several dashboards watch the same
  stream. A "start" while a worker is alive attaches the requester as a
  viewer (status note) instead of starting a second engine; "stop"/"shutdown"
  work from any viewer; the engine stops when the last viewer disconnects.
* The worker runs on a daemon ``threading.Thread`` (MLX inference blocks);
  it broadcasts outbound messages into each viewer's sink (queue + one-slot
  latest-frame / latest-detections buffers) via ``loop.call_soon_threadsafe``.
  Each
  socket's sender task drains its own sink, so receiving controls and
  sending frames run concurrently per viewer.
* The VLM event watch runs on its OWN daemon thread (started/stopped by
  ``set_watch``): it sleeps ``interval_s``, snapshots the worker's latest
  full-resolution frame and asks the configured local VLM whether the
  condition holds. Import of ``vlm_registry`` happens inside the watcher
  thread so this module stays CI-importable; watch failures produce status
  notes (throttled to one per 30s) and a model that cannot load disables the
  watch with a single note.
* Clip capture: the worker keeps a ring buffer of recent encoded JPEGs; when
  a trigger fires it snapshots the pre-roll and accumulates frames for
  ``post_s``. ONE capture occupies the slot across ALL of post-roll
  collection, queued work, and encoding — later triggers still emit their
  events but do not allocate another capture. The clip writer thread starts
  LAZILY when the first completed capture needs encoding (a worker that
  never captures leaks no parked thread) and exits via the ``None`` queue
  sentinel, which every worker exit path sends AFTER any accepted capture
  (FIFO lets it finish); an incomplete (still-collecting) capture is
  discarded on stop. The writer releases the slot under ``_state_lock``
  after encode — success OR failure — then prunes the clips directory to
  the 50 newest files. Decoding + re-encoding the pre-roll blocks for
  seconds and must never run on the frame-processing thread.
* Ask/report evidence: the WS handler captures ONE consistent server-owned
  snapshot — latest full-res frame + detection records + active prompts,
  frame and records under a single ``_frame_lock`` acquisition — BEFORE
  spawning the inference thread; the thread never re-reads live worker
  state. Grounding rule: when prompts are unarmed (paused) or no
  observation exists, the request is rejected with the ``error`` shape
  before the shared inference slot is claimed — no VLM call. An observed
  EMPTY set is a valid observation (the engine ran and saw nothing);
  pausing (``set_prompts([])``) CLEARS the stored evidence so earlier
  counts cannot survive as current observations.
* Box targets (``add_prompt_box``) are ROI labels, not extra inference: the
  installed mlx_vlm build plumbs the ``boxes`` kwarg but never applies box
  conditioning (its geometry encoder is never called), so re-running
  ``predict`` per target only duplicated the main detection at (1 + T)x cost.
  Instead the single main detection pass runs once and every object whose
  center falls inside a drawn ROI is relabeled with that target's label
  ("target N" when the client omitted one). Overlapping ROIs: the first
  matching target wins the label — no double-counting.
* Heavy imports (cv2, PIL, numpy, mlx, mlx_vlm, supervision, sam3_inference)
  happen INSIDE the worker / watcher bodies, so CI — with no mlx and no
  cached weights — can import this module and unit-test the pure helpers.
* ``configure()`` pins an uploads directory and a clips directory; file
  sources resolve ``file_id`` to the SINGLE glob match of ``f"{file_id}*"``
  inside it (mirrors ``web_app._find_upload``). Raw paths are never accepted:
  any path separator or ``..`` in a file_id is rejected up front, and the
  match must resolve inside the configured directory — path traversal is
  impossible.
* Outbound traffic is COALESCED where replacement is safe: binary frames
  AND ``detections`` sets overwrite each other in per-viewer one-slot
  buffers ("latest wins", like the field hub — an empty detection set is a
  REAL value that supersedes older objects, so ``None`` marks an empty
  slot, not an empty set). Events, captures, status, and request-result
  messages (ask_ack / answer / report_result / error) keep their in-order
  queue path and are never dropped. This bounds frames and detections —
  it does not bound every possible outbound source.
* Clip capture writes run on the dedicated writer thread described above —
  never on the read/detect/encode loop.
* Token auth: the endpoint honors ``VB_TOKEN`` via the ``?token=`` query
  parameter (browsers cannot set headers on WebSocket connects). When the
  shared token is enabled, a connect without the correct token is denied
  BEFORE ``accept()`` — a pre-accept close, surfacing as handshake denial
  (close code 4401). Plain ``/api/*`` routes are enforced separately by the
  ``web_app`` HTTP middleware, which deliberately exempts this WebSocket
  route — the door check lives here.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .detection_core import mask_to_polygon
from .inference_admission import AdmissionError, AdmissionHandle, InferenceAdmission

__all__ = [
    "router",
    "configure",
    "resolve_file_id",
    "pack_frame",
    "make_item",
    "validate_control",
    "validate_stream_url",
    "redact_url",
    "validate_zones",
    "validate_box",
    "target_label_at",
    "sanitize_clip_name",
    "RectZone",
    "DwellTracker",
    "DirectionTriggerState",
    "TRIGGER_DEFAULTS",
    "WATCH_DEFAULTS",
]


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers (no cv2 / numpy / mlx — unit-testable anywhere)
# ──────────────────────────────────────────────────────────────────────────────

def pack_frame(frame_id: int, ts_ms: int, jpeg: bytes, telem: dict) -> bytes:
    """Pack one binary wire frame in the field-hub format.

    Layout: ``>III`` (frame_id, timestamp_ms, jpeg_len) + JPEG bytes +
    ``>I`` telem_len + telemetry JSON — note telem_len comes AFTER the JPEG.
    Ints are masked to u32 so long playback sessions or epoch-ms timestamps
    wrap instead of raising.
    """
    telem_bytes = json.dumps(telem, default=str).encode("utf-8")
    header = struct.pack(
        ">III",
        int(frame_id) & 0xFFFFFFFF,
        int(ts_ms) & 0xFFFFFFFF,
        len(jpeg),
    )
    body = jpeg if isinstance(jpeg, bytes) else bytes(jpeg)
    return header + body + struct.pack(">I", len(telem_bytes)) + telem_bytes


def make_item(
    box_px: Sequence[float],
    width: int,
    height: int,
    label: str,
    score: float,
    track_id: int,
    direction: str,
    polygon: Optional[Sequence[Sequence[float]]] = None,
) -> dict:
    """Build one detection item in the client protocol.

    The pixel box is normalized to ``[x1, y1, x2, y2]`` in 0-1 (clamped),
    score is rounded to 3 decimals, and ``color_id`` mirrors ``track_id`` so
    the client palette is stable per track. ``polygon`` (optional) is already
    normalized 0-1 — e.g. ``detection_core.mask_to_polygon`` output — and is
    clamped/rounded through unchanged so clients can paint the object's mask
    outline; clients fall back to the box when absent.
    """
    w = max(1.0, float(width))
    h = max(1.0, float(height))

    def _norm(value: float, span: float) -> float:
        return round(max(0.0, min(1.0, float(value) / span)), 4)

    item = {
        "box": [
            _norm(box_px[0], w),
            _norm(box_px[1], h),
            _norm(box_px[2], w),
            _norm(box_px[3], h),
        ],
        "label": str(label),
        "score": round(float(score), 3),
        "track_id": int(track_id),
        "color_id": int(track_id),
        "direction": str(direction),
    }
    if polygon:
        item["polygon"] = [
            [round(max(0.0, min(1.0, float(p[0]))), 4),
             round(max(0.0, min(1.0, float(p[1]))), 4)]
            for p in polygon if len(p) >= 2
        ]
        if not item["polygon"]:
            del item["polygon"]
    return item


def sanitize_clip_name(name: Any) -> Optional[str]:
    """Return ``name`` when it is a safe clip filename, else None.

    Safe means: only ``[A-Za-z0-9_.-]`` characters (so no path separators of
    either flavor and no traversal) AND a ``.mp4`` extension. Anything else —
    including ``None``, non-strings, ``"../evil"``, ``"a/b.mp4"``,
    ``"x.txt"`` — returns None.
    """
    if not isinstance(name, str):
        return None
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        return None
    if not name.endswith(".mp4"):
        return None
    return name


_URL_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://")
_MAX_URL_LEN = 500


def validate_stream_url(url: Any) -> bool:
    """True when ``url`` is an acceptable network-stream URL.

    Valid means: the scheme (case-insensitive) is one of ``rtsp://``,
    ``rtsps://``, ``http://``, ``https://``; the total length is at most
    500 characters; and whatever follows the scheme is non-empty and
    contains no whitespace. Anything else — plain hostnames without a
    scheme, ``file://``, ``ftp://``, empty strings, non-strings — is False.
    """
    if not isinstance(url, str):
        return False
    if not url or len(url) > _MAX_URL_LEN:
        return False
    lowered = url.lower()
    for scheme in _URL_SCHEMES:
        if lowered.startswith(scheme):
            rest = url[len(scheme):]
            return bool(rest) and re.search(r"\s", rest) is None
    return False


# scheme "://" + userinfo up to the FIRST "@" ("/" and "@" cannot appear
# in userinfo, so the first @ always terminates it).
_USERINFO_RE = re.compile(r"^(?P<head>[A-Za-z][A-Za-z0-9+.\-]*://)(?P<userinfo>[^/@]*)@")


def redact_url(url: str) -> str:
    """Mask userinfo credentials in ``url`` for safe display and logs.

    ``scheme://user:pass@host/…`` becomes ``scheme://user:***@host/…``;
    an empty user masks to ``scheme://***@host/…``. URLs without userinfo
    pass through unchanged (non-strings are returned as-is).
    """
    if not isinstance(url, str):
        return url
    match = _USERINFO_RE.match(url)
    if match is None:
        return url
    head = match.group("head")
    user = match.group("userinfo").split(":", 1)[0]
    tail = url[match.end():]
    return f"{head}{user}:***@{tail}" if user else f"{head}***@{tail}"


def _coord01(value: Any, label: str) -> float:
    """Validate one normalized 0-1 coordinate; raise ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"{label} must be within 0-1")
    return v


def _point01(value: Any, label: str) -> list[float]:
    """Validate a 2-number [x, y] pair in 0-1; raise ValueError otherwise."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must be a 2-number pair")
    return [_coord01(value[0], f"{label}[0]"), _coord01(value[1], f"{label}[1]")]


_MAX_ZONES = 8
_MAX_ZONE_NAME = 40
_MAX_TARGETS = 8
_MAX_TARGET_LABEL = 40


def validate_box(box: Any) -> Optional[list[float]]:
    """Validate a box-prompted target ROI; return floats or None.

    Accepts a list/tuple of exactly 4 real numbers (bools rejected) that are
    normalized 0-1 and ordered ``x1 < x2``, ``y1 < y2`` — inverted boxes are
    rejected, consistent with rect zones. Returns ``[x1, y1, x2, y2]`` as
    floats, else None. Never raises.
    """
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values: list[float] = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        v = float(value)
        if not 0.0 <= v <= 1.0:
            return None
        values.append(v)
    if not values[0] < values[2] or not values[1] < values[3]:
        return None
    return values


def target_label_at(cx: float, cy: float, targets: Sequence[dict]) -> Optional[str]:
    """Return the label of the first target whose rect contains the point.

    Points and target boxes are normalized 0-1; containment is inclusive on
    every edge (mirrors ``RectZone.contains``). Returns None when no target
    matches. Never raises.
    """
    for target in targets:
        box = target.get("box") if isinstance(target, dict) else None
        if not box or len(box) != 4:
            continue
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            return str(target.get("label", ""))
    return None


def validate_zones(zones: Any) -> list[dict]:
    """Validate and normalize a ``set_zones`` payload.

    Each zone must be ``{"kind": "line", "a": [x, y], "b": [x, y]}`` or
    ``{"kind": "rect", "x1", "y1", "x2", "y2"}`` with normalized 0-1
    coordinates (rect corners ordered ``x1 < x2``, ``y1 < y2``) and an
    optional name (string, <= 40 chars; missing/empty names become
    ``"zone N"``). Max 8 zones per set. Returns normalized float dicts;
    raises ``ValueError`` on the first invalid zone.
    """
    if not isinstance(zones, list):
        raise ValueError("zones must be a list")
    if len(zones) > _MAX_ZONES:
        raise ValueError(f"too many zones (max {_MAX_ZONES})")
    out: list[dict] = []
    for i, zone in enumerate(zones):
        if not isinstance(zone, dict):
            raise ValueError(f"zone {i} must be an object")
        kind = zone.get("kind")
        if kind not in ("line", "rect"):
            raise ValueError(f"zone {i}: kind must be 'line' or 'rect'")
        name = zone.get("name")
        if name is None or (isinstance(name, str) and not name.strip()):
            name = f"zone {i + 1}"
        elif not isinstance(name, str):
            raise ValueError(f"zone {i}: name must be a string")
        else:
            name = name.strip()
            if len(name) > _MAX_ZONE_NAME:
                raise ValueError(f"zone {i}: name exceeds {_MAX_ZONE_NAME} chars")
        if kind == "line":
            out.append({
                "kind": "line",
                "name": name,
                "a": _point01(zone.get("a"), f"zone {i}.a"),
                "b": _point01(zone.get("b"), f"zone {i}.b"),
            })
        else:
            x1 = _coord01(zone.get("x1"), f"zone {i}.x1")
            y1 = _coord01(zone.get("y1"), f"zone {i}.y1")
            x2 = _coord01(zone.get("x2"), f"zone {i}.x2")
            y2 = _coord01(zone.get("y2"), f"zone {i}.y2")
            if not x1 < x2 or not y1 < y2:
                raise ValueError(f"zone {i}: rect needs x1<x2 and y1<y2")
            out.append({
                "kind": "rect", "name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return out


class RectZone:
    """Rectangular zone with per-track enter/exit transition detection.

    Coordinates live in whatever space the caller uses (pixels in the
    worker, anything in tests) as long as usage is consistent. ``update``
    fires ``{"zone", "track_id", "detail", "ts", "kind"}`` exactly once per
    enter/exit transition per track.
    """

    def __init__(self, x1: float, y1: float, x2: float, y2: float, name: str) -> None:
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.name = str(name)
        self._inside: set[int] = set()

    def contains(self, cx: float, cy: float) -> bool:
        """True when the point (e.g. a box centroid) is inside the rect."""
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2

    def update(self, track_id: int, inside: bool, ts: float) -> Optional[dict]:
        """Record one observation; return an event dict on a transition."""
        tid = int(track_id)
        was = tid in self._inside
        if bool(inside) == was:
            return None
        if inside:
            self._inside.add(tid)
        else:
            self._inside.discard(tid)
        return {
            "kind": "zone_enter" if inside else "zone_exit",
            "zone": self.name,
            "track_id": tid,
            "detail": f"track {tid} entered" if inside else f"track {tid} exited",
            "ts": float(ts),
        }


class DwellTracker:
    """Fire once per stationary stretch per track after ``dwell_s`` seconds.

    ``update`` starts a per-track timer the first time a track is reported
    stationary, fires once when the stretch reaches ``dwell_s``, and re-arms
    (a fresh stretch may fire again) as soon as the track is reported moving.
    ``dwell_s`` is mutable so a running engine can retune it live.
    """

    def __init__(self, dwell_s: float) -> None:
        self.dwell_s = max(0.0, float(dwell_s))
        self._since: dict[int, float] = {}
        self._fired: set[int] = set()

    def update(self, track_id: int, stationary: bool, ts: float) -> Optional[dict]:
        """Record one observation; return an event dict when dwell elapses."""
        tid = int(track_id)
        if not stationary:
            self._since.pop(tid, None)
            self._fired.discard(tid)
            return None
        if self.dwell_s <= 0 or tid in self._fired:
            return None
        since = self._since.get(tid)
        if since is None:
            self._since[tid] = float(ts)
            return None
        if float(ts) - since >= self.dwell_s:
            self._fired.add(tid)
            self._since.pop(tid, None)
            return {
                "kind": "dwell",
                "track_id": tid,
                "detail": f"track {tid} stationary >= {self.dwell_s:g}s",
                "ts": float(ts),
            }
        return None

    def reset(self) -> None:
        """Drop all per-track dwell state (new scene / file loop)."""
        self._since.clear()
        self._fired.clear()


class DirectionTriggerState:
    """Per-track, per-direction one-shot trigger with re-arm on heading change.

    ``selected`` is the armed direction value: a fixed compass label, or
    ``"any"`` (any real heading — never ``"unknown"``/``"stationary"``), or
    ``"none"`` (never fires). A given (track, direction) pair fires once and
    only re-arms when the track's direction changes.
    """

    _PASSIVE = ("unknown", "stationary")

    def __init__(self) -> None:
        self._last: dict[int, str] = {}
        self._fired: set[tuple[int, str]] = set()

    def update(
        self, track_id: int, direction: str, selected: str, ts: float
    ) -> Optional[dict]:
        """Record one heading observation; return an event dict when it fires."""
        tid = int(track_id)
        d = str(direction)
        if self._last.get(tid) != d:
            # Heading changed (or first sighting) — re-arm this track.
            for key in [k for k in self._fired if k[0] == tid]:
                self._fired.discard(key)
            self._last[tid] = d
        if selected == "none" or d in self._PASSIVE:
            return None
        if selected != "any" and d != selected:
            return None
        if (tid, d) in self._fired:
            return None
        self._fired.add((tid, d))
        return {
            "kind": "direction",
            "track_id": tid,
            "direction": d,
            "detail": f"track {tid} heading {d}",
            "ts": float(ts),
        }

    def reset(self) -> None:
        """Drop all per-track state (new scene / file loop)."""
        self._last.clear()
        self._fired.clear()


_DIRECTION_VALUES = (
    "none", "any",
    "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
)

TRIGGER_DEFAULTS: dict[str, Any] = {
    "line_cross": False,
    "direction": "none",
    "dwell_s": 0.0,
    "clip": True,
    "pre_s": 6.0,
    "post_s": 4.0,
}

WATCH_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "condition": "",
    "interval_s": 4.0,
    "model": "lfm",
}

_WATCH_MODELS = ("lfm", "lfm3b")
_MAX_INTERVAL_S = 30.0
_MIN_INTERVAL_S = 1.0
_MAX_CONDITION = 200

# stream tuning + ask/report limits
_JPEG_QUALITY = 70
_JPEG_QUALITY_MIN = 30
_JPEG_QUALITY_MAX = 95
_MAX_SEND_WIDTH = 1280
_MIN_SEND_WIDTH = 256
_MAX_WIRE_WIDTH = 3840
_MAX_ASK_CHARS = 500
_ENGINE_KEYS = ("sam", "falcon", "lfm")


def _num01_or_more(value: Any, minimum: float) -> bool:
    """True when value is a real number >= minimum (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) >= minimum


def _valid_prompts(value: Any) -> Optional[list[str]]:
    """Return the prompts list when valid, else None.

    Valid means a list of non-empty strings. An EMPTY list is also valid and
    means "detection off" — the hub-protocol pause semantics the Android
    clients rely on (empty set sent explicitly so the server stops).
    """
    if not isinstance(value, list):
        return None
    for prompt in value:
        if not isinstance(prompt, str) or not prompt.strip():
            return None
    return list(value)


def validate_control(msg: Any) -> tuple[str, dict]:
    """Validate one inbound control message.

    Returns ``(action, payload)`` where action is one of ``"start"``,
    ``"set_prompts"``, ``"set_threshold"``, ``"set_stream"``, ``"set_task"``,
    ``"set_engine"``, ``"set_vlm"``, ``"ask"``, ``"report"``,
    ``"set_zones"``, ``"add_prompt_box"``,
    ``"remove_targets"``, ``"set_triggers"``, ``"set_watch"``, ``"stop"``,
    ``"shutdown"``, ``"ignore"`` (protocol hello) — or ``"unknown"`` with an
    empty payload for anything invalid. Control key is ``type`` (hub-protocol
    style) or ``action`` as an alias. Never raises.
    """
    if not isinstance(msg, dict):
        return ("unknown", {})

    # The browser client speaks the hub protocol, which keys controls by
    # "type"; "action" is accepted as an alias for programmatic callers.
    action = msg.get("type") or msg.get("action")

    if action == "hello":
        return ("ignore", {})

    if action == "start":
        source = msg.get("source")
        if source not in ("file", "webcam", "url"):
            return ("unknown", {})
        prompts = _valid_prompts(msg.get("prompts"))
        if prompts is None:
            return ("unknown", {})

        payload: dict[str, Any] = {"source": source, "prompts": prompts}
        if source == "file":
            file_id = msg.get("file_id")
            if not isinstance(file_id, str) or not file_id.strip():
                return ("unknown", {})
            payload["file_id"] = file_id
        elif source == "webcam":
            camera = msg.get("camera")
            if isinstance(camera, bool) or not isinstance(camera, int) or camera < 0:
                return ("unknown", {})
            payload["camera"] = camera
        else:
            url = msg.get("url")
            if not isinstance(url, str) or not validate_stream_url(url):
                return ("unknown", {})
            payload["url"] = url

        # Optional tuning knobs — dropped silently when bogus.
        threshold = msg.get("threshold")
        if (
            isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and 0.0 < float(threshold) < 1.0
        ):
            payload["threshold"] = float(threshold)
        detect_every = msg.get("detect_every")
        if isinstance(detect_every, int) and not isinstance(detect_every, bool) and detect_every >= 1:
            payload["detect_every"] = int(detect_every)
        resolution = msg.get("resolution")
        if isinstance(resolution, int) and not isinstance(resolution, bool) and resolution >= 64:
            payload["resolution"] = int(resolution)
        backbone_every = msg.get("backbone_every")
        if (
            isinstance(backbone_every, int)
            and not isinstance(backbone_every, bool)
            and backbone_every >= 1
        ):
            payload["backbone_every"] = int(backbone_every)
        jpeg_quality = msg.get("jpeg_quality")
        if (
            isinstance(jpeg_quality, int)
            and not isinstance(jpeg_quality, bool)
            and _JPEG_QUALITY_MIN <= jpeg_quality <= _JPEG_QUALITY_MAX
        ):
            payload["jpeg_quality"] = int(jpeg_quality)
        send_width = msg.get("send_width")
        if (
            isinstance(send_width, int)
            and not isinstance(send_width, bool)
            and _MIN_SEND_WIDTH <= send_width <= _MAX_WIRE_WIDTH
        ):
            payload["send_width"] = int(send_width)
        if msg.get("task") in ("segment", "detect"):
            payload["task"] = msg["task"]
        return ("start", payload)

    if action == "set_prompts":
        prompts = _valid_prompts(msg.get("prompts"))
        if prompts is None:
            return ("unknown", {})
        return ("set_prompts", {"prompts": prompts})

    if action == "set_zones":
        if not isinstance(msg.get("zones"), list):
            return ("unknown", {})
        try:
            zones = validate_zones(msg["zones"])
        except ValueError:
            return ("unknown", {})
        return ("set_zones", {"zones": zones})

    if action == "add_prompt_box":
        box = validate_box(msg.get("box"))
        if box is None:
            return ("unknown", {})
        label = msg.get("label")
        if label is None or (isinstance(label, str) and not label.strip()):
            label = None  # engine assigns the default "target N"
        elif not isinstance(label, str) or len(label.strip()) > _MAX_TARGET_LABEL:
            return ("unknown", {})
        else:
            label = label.strip()
        # Cap the staged targets (the running cap lives on the worker).
        if len(_pending_targets or []) >= _MAX_TARGETS:
            return ("unknown", {})
        return ("add_prompt_box", {"box": box, "label": label})

    if action == "remove_targets":
        return ("remove_targets", {})

    if action == "set_triggers":
        payload = {}
        if "line_cross" in msg:
            if not isinstance(msg["line_cross"], bool):
                return ("unknown", {})
            payload["line_cross"] = msg["line_cross"]
        if "direction" in msg:
            if msg["direction"] not in _DIRECTION_VALUES:
                return ("unknown", {})
            payload["direction"] = msg["direction"]
        if "dwell_s" in msg:
            if not _num01_or_more(msg["dwell_s"], 0.0):
                return ("unknown", {})
            payload["dwell_s"] = float(msg["dwell_s"])
        if "clip" in msg:
            if not isinstance(msg["clip"], bool):
                return ("unknown", {})
            payload["clip"] = msg["clip"]
        if "pre_s" in msg:
            if not _num01_or_more(msg["pre_s"], 0.0):
                return ("unknown", {})
            payload["pre_s"] = float(msg["pre_s"])
        if "post_s" in msg:
            if not _num01_or_more(msg["post_s"], 0.0):
                return ("unknown", {})
            payload["post_s"] = float(msg["post_s"])
        return ("set_triggers", payload)

    if action == "set_watch":
        payload = {}
        if "enabled" in msg:
            if not isinstance(msg["enabled"], bool):
                return ("unknown", {})
            payload["enabled"] = msg["enabled"]
        if "condition" in msg:
            condition = msg["condition"]
            if not isinstance(condition, str) or len(condition) > _MAX_CONDITION:
                return ("unknown", {})
            payload["condition"] = condition
        if "interval_s" in msg:
            value = msg["interval_s"]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not _MIN_INTERVAL_S <= float(value) <= _MAX_INTERVAL_S
            ):
                return ("unknown", {})
            payload["interval_s"] = float(value)
        if "model" in msg:
            if msg["model"] not in _WATCH_MODELS:
                return ("unknown", {})
            payload["model"] = msg["model"]
        return ("set_watch", payload)

    if action == "set_threshold":
        value = msg.get("threshold")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) < 1.0
        ):
            return ("unknown", {})
        return ("set_threshold", {"threshold": float(value)})

    if action == "set_stream":
        payload = {}
        if "jpeg_quality" in msg:
            quality = msg["jpeg_quality"]
            if (
                isinstance(quality, bool)
                or not isinstance(quality, int)
                or not _JPEG_QUALITY_MIN <= quality <= _JPEG_QUALITY_MAX
            ):
                return ("unknown", {})
            payload["jpeg_quality"] = int(quality)
        if "send_width" in msg:
            width = msg["send_width"]
            if (
                isinstance(width, bool)
                or not isinstance(width, int)
                or not _MIN_SEND_WIDTH <= width <= _MAX_WIRE_WIDTH
            ):
                return ("unknown", {})
            payload["send_width"] = int(width)
        if not payload:
            return ("unknown", {})
        return ("set_stream", payload)

    if action == "set_task":
        if msg.get("task") not in ("segment", "detect"):
            return ("unknown", {})
        return ("set_task", {"task": msg["task"]})

    if action == "set_engine":
        payload = {}
        for key in _ENGINE_KEYS:
            if key in msg:
                if not isinstance(msg[key], bool):
                    return ("unknown", {})
                payload[key] = msg[key]
        if "lfm_model" in msg:
            if msg["lfm_model"] not in _WATCH_MODELS:
                return ("unknown", {})
            payload["lfm_model"] = msg["lfm_model"]
        return ("set_engine", payload)

    if action == "set_vlm":
        model = msg.get("model")
        if not isinstance(model, str) or not model.strip():
            return ("unknown", {})
        return ("set_vlm", {"model": model.strip()})

    if action == "ask":
        question = msg.get("question")
        if not isinstance(question, str) or not question.strip():
            return ("unknown", {})
        if len(question) > _MAX_ASK_CHARS:
            return ("unknown", {})
        return ("ask", {"question": question.strip()})

    if action == "report":
        payload = {"summary": "", "report_type": "field"}
        summary = msg.get("summary")
        if summary is not None:
            if not isinstance(summary, str) or len(summary) > _MAX_ASK_CHARS:
                return ("unknown", {})
            payload["summary"] = summary.strip()
        report_type = msg.get("report_type")
        if report_type is not None:
            if not isinstance(report_type, str) or not report_type.strip():
                return ("unknown", {})
            if len(report_type) > 40:
                return ("unknown", {})
            payload["report_type"] = report_type.strip()
        return ("report", payload)

    if action in ("stop", "shutdown"):
        return (action, {})

    return ("unknown", {})


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (uploads + clips dirs), file_id resolution with traversal guard
# ──────────────────────────────────────────────────────────────────────────────

_uploads_dir: Optional[Path] = None
_clips_dir: Optional[Path] = None


def configure(uploads_dir: Path, clips_dir: Optional[Path] = None) -> None:
    """Pin the uploads and clips directories.

    Must be called (typically from web_app startup with its UPLOADS and CLIPS
    dirs) before file-based starts will resolve and before clip capture will
    write; webcam starts need no config. ``clips_dir`` is created when
    missing; when it is left as None, clip capture stays disabled.
    """
    global _uploads_dir, _clips_dir
    _uploads_dir = Path(uploads_dir)
    if clips_dir is None:
        _clips_dir = None
    else:
        _clips_dir = Path(clips_dir)
        try:
            _clips_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            _clips_dir = None


def resolve_file_id(file_id: str) -> Optional[Path]:
    """Resolve a file_id to the SINGLE glob match inside the uploads dir.

    Mirrors ``web_app._find_upload``: uploads are stored as
    ``f"{file_id}{ext}"`` so ``f"{file_id}*"`` must match exactly one file.
    Returns None when unconfigured, on any path-separator / ``..`` in the
    file_id (traversal guard), on zero or multiple matches, or when the
    match escapes the configured directory. Never accepts arbitrary paths.
    """
    base = _uploads_dir
    if base is None or not file_id:
        return None
    if "/" in file_id or "\\" in file_id or ".." in file_id:
        return None
    try:
        matches = [p for p in base.glob(f"{file_id}*") if p.is_file()]
    except (OSError, ValueError):
        return None
    if len(matches) != 1:
        return None
    resolved = matches[0].resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError:
        return None
    return resolved


# ──────────────────────────────────────────────────────────────────────────────
# VLM event watch — one daemon watcher thread per process
# ──────────────────────────────────────────────────────────────────────────────

_WATCH_NOTE_THROTTLE_S = 30.0

_watch_lock = threading.Lock()
_watch_cfg: dict[str, Any] = dict(WATCH_DEFAULTS)
_watch_thread: Optional[threading.Thread] = None
_watch_stop: Optional[threading.Event] = None
_watch_busy = False          # an ask is in flight — skip ticks until done
_watch_model_ok = False      # current model answered at least once
_watch_last_note = 0.0       # monotonic ts of the last error status note


def _watch_loop(stop_event: threading.Event, worker: "_EngineWorker") -> None:
    """Watcher body: periodically ask the VLM whether the condition holds.

    Skips ticks while the engine worker is not running or a previous ask is
    still in flight. Failures become throttled status notes; a model that
    never answers (cannot load) disables the watch with one note.
    """
    global _watch_busy, _watch_model_ok, _watch_cfg, _watch_last_note
    while not stop_event.is_set():
        with _watch_lock:
            cfg = dict(_watch_cfg)
        interval = max(_MIN_INTERVAL_S, float(cfg.get("interval_s", 4.0)))
        if stop_event.wait(interval):
            break

        if not _alive(worker) or worker.stop_event.is_set():
            break
        condition = str(cfg.get("condition") or "").strip()
        if not condition:
            continue
        with _watch_lock:
            if _watch_busy:
                continue
            _watch_busy = True
        if not worker._register_native_job():
            with _watch_lock:
                _watch_busy = False
            break
        try:
            # Imported here — inside the watcher thread — so this module
            # stays importable on hosts with no mlx stack at all.
            from . import vlm_registry

            snap = worker.get_latest_frame()
            if snap is None:
                continue
            frame_id, image = snap
            try:
                vlm_registry.set_model(str(cfg.get("model", "lfm")))
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            reply = vlm_registry.ask(
                "You are a video event watcher. Answer with exactly YES or NO. "
                "Condition: " + condition,
                image=image,
            )
            with _watch_lock:
                _watch_model_ok = True
            first_word = str(reply).strip().split()[:1]
            if first_word and first_word[0].strip(".!,;:").lower() == "yes":
                ts = time.time()
                worker.request_capture("watch", ts)
                worker.push({
                    "type": "event",
                    "event": {
                        "kind": "watch",
                        "zone": "",
                        "direction": "",
                        "track_id": None,
                        "ts": float(ts),
                        "frame_id": int(frame_id),
                        "detail": condition,
                    },
                })
        except Exception as exc:  # noqa: BLE001 — the watcher must keep running
            with _watch_lock:
                ever_loaded = _watch_model_ok
                now = time.monotonic()
                throttled = (now - _watch_last_note) < _WATCH_NOTE_THROTTLE_S
                if not throttled:
                    _watch_last_note = now
            if not ever_loaded:
                # The model never answered once — treat as unloadable and stop.
                worker.push({"type": "status", "note": f"watch disabled: {exc}"})
                with _watch_lock:
                    _watch_cfg = dict(_watch_cfg, enabled=False)
                break
            if not throttled:
                worker.push({"type": "status", "note": f"watch error: {exc}"})
        finally:
            try:
                worker._finish_native_job()
            finally:
                with _watch_lock:
                    _watch_busy = False


# ── Ask / report — one in-flight slot shared by both (Android semantics) ─────
_ask_lock = threading.Lock()
_ask_busy = False


def _claim_ask_slot() -> bool:
    """Claim the shared ask/report slot; False when one is already running."""
    global _ask_busy
    with _ask_lock:
        if _ask_busy:
            return False
        _ask_busy = True
        return True


def _release_ask_slot() -> None:
    global _ask_busy
    with _ask_lock:
        _ask_busy = False


def _vlm_label() -> str:
    """MODELS key of the selected ask VLM (or ``"custom"`` for a raw pin)."""
    try:
        from . import vlm_registry

        return vlm_registry.current_key() or "custom"
    except Exception:  # noqa: BLE001 — label only
        return "custom"


def _counts_summary(records: Sequence[dict]) -> str:
    """Build the report counts line SERVER-SIDE from detection records.

    e.g. ``"3x car, 1x person"`` (first-appearance order), or
    ``"no objects observed"`` for an empty set. The browser-supplied
    ``summary`` is accepted for wire compatibility but deliberately never
    forwarded to the model — a forged summary must not change the counts.
    """
    counts: dict[str, int] = {}
    for rec in records:
        label = str(rec.get("label", "object")) or "object"
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        return "no objects observed"
    return ", ".join(f"{n}x {label}" for label, n in counts.items())


def _ask_once(worker: "_EngineWorker", question: str, snapshot: dict) -> None:
    """Answer one question about the captured evidence snapshot.

    Runs on a background thread: the VLM generate blocks for seconds (MLX) —
    never the event loop. ``snapshot`` is the consistent (frame, records,
    prompts) triple captured at dispatch time; this thread must NOT re-read
    live worker state. Failures push the pinned ``error`` shape so the
    client clears its request UI.
    """
    try:
        from . import vlm_registry

        reply = vlm_registry.ask(
            question,
            detections=snapshot["records"],
            prompts=snapshot["prompts"],
            image=snapshot["frame"],
        )
        worker.push({"type": "answer", "answer": str(reply)})
    except Exception as exc:  # noqa: BLE001 — report through the socket
        worker.push({"type": "error", "error": f"ask failed: {exc}"})
    finally:
        try:
            _release_ask_slot()
        finally:
            worker._finish_native_job()


def _report_once(worker: "_EngineWorker", snapshot: dict, report_type: str) -> None:
    """Write one grounded field report from the evidence snapshot.

    The counts line is derived server-side from the snapshot's detection
    records (see :func:`_counts_summary`) — never from the browser's
    ``summary``. Failures push the pinned ``error`` shape.
    """
    try:
        from . import vlm_registry

        summary = _counts_summary(snapshot["records"])
        reply = vlm_registry.generate_report(
            summary, report_type=report_type, image=snapshot["frame"]
        )
        worker.push({"type": "report_result", "text": str(reply)})
    except Exception as exc:  # noqa: BLE001 — report through the socket
        worker.push({"type": "error", "error": f"report failed: {exc}"})
    finally:
        try:
            _release_ask_slot()
        finally:
            worker._finish_native_job()


async def _apply_watch(
    payload: dict, worker: Optional["_EngineWorker"] = None
) -> bool:
    """Merge a ``set_watch`` payload into the watch config; start/stop the thread.

    Returns the resulting enabled flag. Joining the old watcher happens off
    the event loop thread (``asyncio.to_thread``) — never block the loop.
    """
    global _watch_thread, _watch_stop, _watch_model_ok, _watch_cfg
    with _watch_lock:
        merged = dict(WATCH_DEFAULTS)
        merged.update(_watch_cfg)
        merged.update(payload)
        if "model" in payload and payload["model"] != _watch_cfg.get("model"):
            _watch_model_ok = False
        _watch_cfg = merged
        enabled = bool(merged["enabled"])

    if enabled:
        if not _alive(worker) or worker._is_stopping():
            with _watch_lock:
                _watch_cfg = dict(_watch_cfg, enabled=False)
            return False
        start_error: Optional[Exception] = None
        with _watch_lock:
            if _watch_thread is None or not _watch_thread.is_alive():
                _watch_stop = threading.Event()
                worker._watch_stop = _watch_stop
                _watch_thread = threading.Thread(
                    target=_watch_loop, args=(_watch_stop, worker),
                    name="live-watch", daemon=True,
                )
                try:
                    _watch_thread.start()
                except Exception as exc:  # noqa: BLE001 — report through status
                    start_error = exc
                    _watch_stop = None
                    _watch_thread = None
                    worker._watch_stop = None
                    _watch_cfg = dict(_watch_cfg, enabled=False)
        if start_error is not None:
            worker.push({"type": "status", "note": f"watch failed to start: {start_error}"})
            return False
        return True

    with _watch_lock:
        stop = _watch_stop
    if stop is not None:
        stop.set()
        thread = _watch_thread
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 2.0)
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Worker — one running engine (thread + outbound queue + live prompts)
# ──────────────────────────────────────────────────────────────────────────────

def _records_from_items(items: Sequence[dict]) -> list[dict]:
    """Compact VLM-facing ``{label, score, centroid_norm, source}`` records.

    The shape ``vlm_registry.ask`` formats into grounded prompt sections,
    derived from one detect frame's wire items.
    """
    records: list[dict] = []
    for item in items:
        box = item.get("box") or [0.5, 0.5, 0.5, 0.5]
        records.append({
            "label": str(item.get("label", "object")),
            "score": float(item.get("score", 0.0)),
            "centroid_norm": {
                "x": (float(box[0]) + float(box[2])) / 2.0,
                "y": (float(box[1]) + float(box[3])) / 2.0,
            },
            "source": "sam",
        })
    return records

_MAX_SEND_WIDTH = 1280
_JPEG_QUALITY = 70
_HELD_EVERY = 5  # resend held boxes every Nth non-detect frame
_MAX_RING_FRAMES = 150
_MAX_CLIPS = 50
_MAX_STREAM_READ_FAILS = 40  # consecutive url read failures before giving up
_ENGINE_JOIN_TIMEOUT_S = 10.0
# Worker-thread cap on joining its clip writer at exit. Must stay below the
# event loop's 10s worker join so a hung encode cannot stall the loop's own
# cleanup for long — the writer is a daemon and the process never cancels it.
_CAPTURE_JOIN_TIMEOUT_S = 8.0

# One-slot marker meaning "a wire frame is waiting in the sink's pending
# slot" — lets each viewer's sender ship only the newest queued frame
# (latest wins), per viewer.
_FRAME_SENTINEL = object()
# Same pattern for ``detections`` messages: the marker means "a detection
# set is pending"; the slot itself holds the newest set, INCLUDING empty
# ones (None — not [] — means nothing pending).
_DETECTIONS_SENTINEL = object()


class _ClientSink:
    """One attached viewer's outbound path.

    A queue for in-order JSON messages (events / captures / status /
    request-result — never dropped) plus one-slot coalescing buffers for
    the two replaceable streams: wire frames and ``detections`` sets
    (latest-wins, so a slow viewer never grows a stale backlog of either).
    Every connected socket gets a sink and the running engine broadcasts
    to all of them — that is what lets several dashboards watch the same
    stream. This bounds frames and detections specifically, not every
    possible outbound source.
    """

    __slots__ = (
        "loop", "queue", "out_lock",
        "pending_frame", "pending_detections",
    )

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.queue: "asyncio.Queue[Any]" = asyncio.Queue()
        self.out_lock = threading.Lock()
        self.pending_frame: Optional[bytes] = None
        self.pending_detections: Optional[list] = None

    def push(self, message: Any) -> None:
        """Queue one JSON message; a dead loop just drops this sink."""
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, message)
        except RuntimeError:
            pass  # event loop gone (client disconnected)

    def push_frame(self, packed: bytes) -> None:
        """Accept one wire frame, coalescing to the newest per viewer."""
        with self.out_lock:
            first = self.pending_frame is None
            self.pending_frame = packed
        if first:
            self.push(_FRAME_SENTINEL)

    def push_detections(self, items: list) -> None:
        """Accept one detection set, coalescing to the newest per viewer.

        An EMPTY set is a real value (it tells the client to drop stale
        boxes) and must supersede older non-empty ones — so ``None``, not
        ``[]``, marks an empty slot.
        """
        with self.out_lock:
            first = self.pending_detections is None
            self.pending_detections = list(items)
        if first:
            self.push(_DETECTIONS_SENTINEL)


class _EngineWorker:
    """A single running engine: worker thread state plus tunables.

    Holds the stop event, the viewer sink set it broadcasts to, and the
    current prompts under a small lock so ``set_prompts`` can retarget
    detection mid-run from the WS thread. Also owns the shared smart-capture
    state (zones, box-prompted targets, triggers, ring buffer, pending
    capture, latest full frame) guarded by ``_state_lock`` / ``_frame_lock``.
    """

    def __init__(
        self,
        cfg: dict,
        sinks: "set[_ClientSink]",
    ) -> None:
        self.cfg = cfg
        self.sinks = sinks
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self._prompts_lock = threading.Lock()
        self._prompts: list[str] = list(cfg["prompts"])

        # ── smart-capture shared state ────────────────────────────────────
        self._state_lock = threading.Lock()
        self._zones: list[dict] = []            # normalized zone specs
        self._zones_dirty = False
        self._rect_zones: list[RectZone] = []   # rebuilt in pixel coords
        self._line_counters: dict[str, Any] = {}   # name -> zones.LineZoneCounter
        self._line_totals: dict[str, int] = {}
        self._targets: list[dict] = []   # box-prompted targets (normalized)
        self._target_seq = 0             # running per-engine target sequence
        self._triggers: dict[str, Any] = dict(TRIGGER_DEFAULTS)
        self._ring: deque = deque()
        self._capture: Optional[dict] = None
        self._fps = 30.0
        self._dims: tuple[int, int] = (0, 0)
        self._dir_state = DirectionTriggerState()
        self._dwell = DwellTracker(0.0)

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[tuple[int, Any]] = None
        self._last_frame_items: list[dict] = []   # detection items for that frame

        # ── live-tunable knobs (start keys, mutable via set_* controls) ───
        # NOTE: the legacy "backbone_every" start key is validated for
        # compatibility but deliberately NOT stored — image features are
        # recomputed on every detection pass (see the module docstring).
        self._tuning_lock = threading.Lock()
        self._threshold = min(0.99, max(0.01, float(cfg.get("threshold", 0.15))))
        self._task = str(cfg.get("task", "segment"))
        self._jpeg_quality = max(
            _JPEG_QUALITY_MIN,
            min(_JPEG_QUALITY_MAX, int(cfg.get("jpeg_quality", _JPEG_QUALITY))),
        )
        self._send_width = max(
            _MIN_SEND_WIDTH,
            min(_MAX_WIRE_WIDTH, int(cfg.get("send_width", _MAX_SEND_WIDTH))),
        )

        # ── off-loop clip writer (started LAZILY on first capture) ────────
        # The writer thread is NOT created here: a worker that never
        # captures must not leak a parked thread. ``_ensure_capture_writer``
        # starts it when a completed capture needs encoding, and every
        # worker exit path signals shutdown with the None sentinel (see
        # ``_shutdown_capture_writer``). The slot bookkeeping guarantees at
        # most one completed capture is ever queued, so the queue cannot
        # grow during a slow encode.
        self._capture_queue: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._capture_thread: Optional[threading.Thread] = None
        self._native_condition = threading.Condition()
        self._native_job_count = 0
        self._stopping = False
        self._admission_handle: Optional[AdmissionHandle] = None
        self._watch_stop: Optional[threading.Event] = None

    def _register_native_job(self) -> bool:
        """Accept one Ask, Report, or Watch call before it can start."""
        with self._native_condition:
            if (
                self._stopping
                or self.stop_event.is_set()
                or self._admission_handle is None
            ):
                return False
            self._native_job_count += 1
            return True

    def _finish_native_job(self) -> None:
        """Release one accepted native call and wake a draining worker."""
        with self._native_condition:
            self._native_job_count -= 1
            if self._native_job_count == 0:
                self._native_condition.notify_all()

    def _is_stopping(self) -> bool:
        with self._native_condition:
            return self._stopping

    def _begin_stopping(self) -> None:
        with self._native_condition:
            self._stopping = True
        self.stop_event.set()
        if self._watch_stop is not None:
            self._watch_stop.set()

    def _finish_native_owner(self) -> None:
        """Hold admission through detector and accepted native job drain."""
        self._begin_stopping()
        with self._native_condition:
            while self._native_job_count:
                self._native_condition.wait()
            handle = self._admission_handle
            self._admission_handle = None
        if handle is not None:
            handle.release()

    def _thread_entry(self) -> None:
        """Wrap the detector entry so its admission survives accepted jobs."""
        try:
            with self._native_condition:
                run_native = (
                    not self._stopping
                    and not self.stop_event.is_set()
                    and self._admission_handle is not None
                )
            if run_native:
                self.run()
            else:
                self._shutdown_capture_writer()
        finally:
            self._finish_native_owner()
            self.push({"type": "engine_stopped"})

    def get_prompts(self) -> list[str]:
        """Return a copy of the current prompts (thread-safe)."""
        with self._prompts_lock:
            return list(self._prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        """Replace the active prompts (thread-safe; picked up next frame).

        An empty list is "detection off": the worker skips the model until
        new prompts arrive. Pausing ALSO clears the stored evidence (latest
        frame + detection records) so earlier counts cannot survive as
        current observations for ask/report — the next real detect pass
        re-establishes it.
        """
        with self._prompts_lock:
            self._prompts = list(prompts)
        if not prompts:
            with self._frame_lock:
                self._latest_frame = None
                self._last_frame_items = []

    def set_threshold(self, value: float) -> None:
        """Retune the detection threshold live (thread-safe)."""
        with self._tuning_lock:
            self._threshold = min(0.99, max(0.01, float(value)))

    def set_task(self, task: str) -> None:
        """Switch segment/detect live ("detect" skips polygon emission)."""
        with self._tuning_lock:
            self._task = str(task)

    def set_stream(self, jpeg_quality: Optional[int] = None,
                   send_width: Optional[int] = None) -> None:
        """Retune the outbound JPEG encode live (thread-safe)."""
        with self._tuning_lock:
            if jpeg_quality is not None:
                self._jpeg_quality = max(
                    _JPEG_QUALITY_MIN, min(_JPEG_QUALITY_MAX, int(jpeg_quality))
                )
            if send_width is not None:
                self._send_width = max(
                    _MIN_SEND_WIDTH, min(_MAX_WIRE_WIDTH, int(send_width))
                )

    def set_zones(self, zones: list[dict]) -> None:
        """Replace the zone set (thread-safe; rebuilt on the next detect frame)."""
        with self._state_lock:
            self._zones = list(zones)
            self._zones_dirty = True

    def add_target(self, box: Sequence[float], label: Optional[str] = None) -> Optional[int]:
        """Add a box-prompted target; return its sequence number, or None when full.

        The box is normalized 0-1 xyxy (already validated by
        ``validate_box``). When ``label`` is missing/empty the engine assigns
        the default ``"target N"`` from its running per-engine sequence, which
        is never reset by ``clear_targets``. Thread-safe; picked up on the
        next detect frame.
        """
        with self._state_lock:
            if len(self._targets) >= _MAX_TARGETS:
                return None
            self._target_seq += 1
            resolved = label if label else f"target {self._target_seq}"
            self._targets.append({
                "box": [float(c) for c in box],
                "label": str(resolved),
            })
            return self._target_seq

    def clear_targets(self) -> None:
        """Drop every box-prompted target (thread-safe; keeps the sequence)."""
        with self._state_lock:
            self._targets = []

    def get_targets(self) -> list[dict]:
        """Return a copy of the active box-prompted targets (thread-safe)."""
        with self._state_lock:
            return [dict(t, box=list(t["box"])) for t in self._targets]

    def last_detection_records(self) -> list[dict]:
        """VLM-facing detection records for the latest detect frame.

        Compact ``{label, score, centroid_norm, source}`` dicts — the shape
        ``vlm_registry.ask`` formats into grounded prompt sections.
        """
        with self._frame_lock:
            return _records_from_items(self._last_frame_items)

    def evidence_snapshot(self) -> Optional[dict]:
        """One consistent server-owned snapshot for ask/report dispatch.

        Returns ``{"frame_id", "frame", "records", "prompts"}`` where frame
        and detection records come from a SINGLE ``_frame_lock`` acquisition
        (they are always stored together by the detect pass) and ``prompts``
        is the active prompt list. The WS handler calls this BEFORE spawning
        the inference thread; the thread must not re-read live worker state.

        Returns None when no current observation exists — i.e. the stored
        evidence was cleared by a pause (``set_prompts([])``) or no real
        detect pass has run yet. An observed EMPTY set is NOT None: the
        engine ran and saw nothing, which is a valid observation.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame_id, image = self._latest_frame
            records = _records_from_items(self._last_frame_items)
        return {
            "frame_id": int(frame_id),
            "frame": image,
            "records": records,
            "prompts": self.get_prompts(),
        }

    def set_triggers(self, partial: dict) -> None:
        """Merge trigger keys into the current config (thread-safe)."""
        with self._state_lock:
            merged = dict(self._triggers)
            merged.update(partial)
            self._triggers = merged
            self._dwell.dwell_s = max(0.0, float(merged.get("dwell_s", 0.0) or 0.0))

    def get_latest_frame(self) -> Optional[tuple[int, Any]]:
        """Return ``(frame_id, full-res frame)`` of the latest detect frame.

        The frame is the same full-resolution object the worker already
        builds before its JPEG downscale (RGB PIL image); treat it as
        read-only. None until the first detect frame.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            frame_id, image = self._latest_frame
            return int(frame_id), image

    def request_capture(self, kind: str, ts: float) -> bool:
        """Claim the capture slot for one clip; False when it is occupied.

        The slot spans ALL of post-roll collection, queued work, and
        encoding: while one capture is alive (any state) later triggers
        still emit their events but do NOT allocate another capture. The
        snapshot of the current ring buffer is the pre-roll; the worker
        accumulates frames until ``trigger_ts + post_s`` and then hands the
        capture to the clip writer, which releases the slot after encode.
        """
        if _clips_dir is None:
            return False
        with self._state_lock:
            if not bool(self._triggers.get("clip", True)):
                return False
            if self._capture is not None:
                return False
            post_s = max(0.0, float(self._triggers.get("post_s", 4.0) or 0.0))
            self._capture = {
                "kind": str(kind),
                "trigger_ts": float(ts),
                "deadline": float(ts) + post_s,
                "frames": list(self._ring),
                # "collecting" -> post-roll growing; "encoding" -> handed to
                # the writer, slot still claimed until it finishes.
                "state": "collecting",
            }
            return True

    def push(self, message: Any) -> None:
        """Broadcast one outbound message (bytes or dict) to every viewer."""
        for sink in tuple(self.sinks):
            sink.push(message)

    def push_frame(self, packed: bytes) -> None:
        """Broadcast one wire frame with latest-frame-wins coalescing.

        A slow dashboard must not build an unbounded backlog of stale JPEGs
        (memory on a 16GB field box, plus video latency climbing past the
        3s auto-hide contract): each viewer's sink keeps only the newest
        frame and its sender ships that. In-order JSON messages travel via
        :meth:`push` and are never dropped.
        """
        for sink in tuple(self.sinks):
            sink.push_frame(packed)

    def push_detections(self, items: list[dict]) -> None:
        """Broadcast one detection set with latest-wins coalescing.

        Same replaceable-stream reasoning as :meth:`push_frame`: a slow
        viewer must not accumulate old detection snapshots behind current
        video, so each sink keeps only the newest set — INCLUDING empty
        ones (they tell the client to drop stale boxes). Events, captures,
        status, and request-result messages keep their in-order path via
        :meth:`push` and are never dropped.
        """
        for sink in tuple(self.sinks):
            sink.push_detections(items)

    # ── zone / trigger evaluation ─────────────────────────────────────────

    def _rebuild_zones_locked(self) -> None:
        """Materialize normalized zone specs into pixel-space objects.

        Called under ``_state_lock`` from the worker thread only (supervision
        is imported lazily — it is a declared dependency). Also resets the
        line-cross counters, per "zones changed" semantics.
        """
        self._rect_zones = []
        self._line_counters = {}
        self._line_totals = {}
        src_w, src_h = self._dims
        if src_w <= 0 or src_h <= 0 or not self._zones:
            return
        from .zones import LineZoneCounter

        for zone in self._zones:
            if zone["kind"] == "rect":
                self._rect_zones.append(RectZone(
                    zone["x1"] * src_w, zone["y1"] * src_h,
                    zone["x2"] * src_w, zone["y2"] * src_h,
                    zone["name"],
                ))
            else:
                ax, ay = zone["a"]
                bx, by = zone["b"]
                self._line_counters[zone["name"]] = LineZoneCounter(
                    start=(int(ax * src_w), int(ay * src_h)),
                    end=(int(bx * src_w), int(by * src_h)),
                )

    def _evaluate_triggers(self, items: list[dict], ts: float) -> list[dict]:
        """Run every armed trigger over one detect frame's items.

        Returns raw event dicts (kind/zone/direction/track_id/detail/ts);
        the caller wraps them into the wire envelope and requests captures.
        """
        with self._state_lock:
            if self._zones_dirty:
                self._zones_dirty = False
                self._rebuild_zones_locked()
            rect_zones = list(self._rect_zones)
            line_counters = dict(self._line_counters)
            line_totals = dict(self._line_totals)
            triggers = dict(self._triggers)

        events: list[dict] = []
        if not items:
            return events
        src_w, src_h = self._dims

        # ── line crossings (supervision LineZone over pixel boxes) ────────
        if triggers.get("line_cross") and line_counters:
            import numpy as np
            import supervision as sv

            xyxy = []
            tids = []
            for it in items:
                box = it.get("box") or [0.0, 0.0, 0.0, 0.0]
                xyxy.append([
                    float(box[0]) * src_w, float(box[1]) * src_h,
                    float(box[2]) * src_w, float(box[3]) * src_h,
                ])
                tids.append(int(it.get("track_id", 0) or 0))
            dets = sv.Detections(
                xyxy=np.asarray(xyxy, dtype=np.float64),
                tracker_id=np.asarray(tids, dtype=np.int64),
            )
            new_totals = dict(line_totals)
            for name, counter in line_counters.items():
                res = counter.update(dets)
                for tid in res.get("crossed_in", []):
                    events.append({
                        "kind": "line_cross", "zone": name, "track_id": int(tid),
                        "detail": f"track crossed {name} in", "ts": ts,
                    })
                for tid in res.get("crossed_out", []):
                    events.append({
                        "kind": "line_cross", "zone": name, "track_id": int(tid),
                        "detail": f"track crossed {name} out", "ts": ts,
                    })
                total = int(res.get("in", 0)) + int(res.get("out", 0))
                identified = len(res.get("crossed_in", [])) + len(res.get("crossed_out", []))
                for _ in range(max(0, total - line_totals.get(name, 0) - identified)):
                    events.append({
                        "kind": "line_cross", "zone": name, "track_id": None,
                        "detail": f"track crossed {name}", "ts": ts,
                    })
                new_totals[name] = total
            with self._state_lock:
                self._line_totals = new_totals

        # ── rect enter/exit (always on while rect zones exist) ────────────
        for it in items:
            tid = it.get("track_id")
            if tid is None:
                continue
            box = it.get("box") or [0.0, 0.0, 0.0, 0.0]
            cx = (float(box[0]) + float(box[2])) / 2.0 * src_w
            cy = (float(box[1]) + float(box[3])) / 2.0 * src_h
            for rect in rect_zones:
                ev = rect.update(int(tid), rect.contains(cx, cy), ts)
                if ev:
                    events.append(ev)

        # ── direction trigger ─────────────────────────────────────────────
        selected = str(triggers.get("direction", "none"))
        if selected != "none":
            for it in items:
                ev = self._dir_state.update(
                    it.get("track_id", 0),
                    str(it.get("direction", "unknown")),
                    selected, ts,
                )
                if ev:
                    events.append(ev)

        # ── dwell trigger ─────────────────────────────────────────────────
        dwell_s = float(triggers.get("dwell_s", 0.0) or 0.0)
        if dwell_s > 0:
            self._dwell.dwell_s = dwell_s
            for it in items:
                ev = self._dwell.update(
                    it.get("track_id", 0),
                    str(it.get("direction")) == "stationary",
                    ts,
                )
                if ev:
                    events.append(ev)
        return events

    # ── clip writing ──────────────────────────────────────────────────────

    def _handoff_capture(self) -> Optional[dict]:
        """Mark the pending capture ready for the writer (worker thread).

        The slot STAYS claimed — the capture dict remains ``self._capture``
        with ``state`` advanced to ``"encoding"`` — until the writer
        releases it after encode (success or failure), so no second capture
        can start while one is queued or being written. Only a
        ``"collecting"`` capture transitions; repeats are no-ops.
        """
        with self._state_lock:
            capture = self._capture
            if capture is None or capture.get("state") != "collecting":
                return None
            capture["state"] = "encoding"
        return capture

    def _dispatch_due_capture(self) -> None:
        """Hand a completed capture to the clip writer (worker thread).

        Called from the frame loop when the post-roll deadline passes. This
        is where the writer thread is lazily started — a worker that never
        captures never pays for one. At most ONE completed capture can be
        queued: the slot is claimed until the writer finishes encoding.
        """
        capture = self._handoff_capture()
        if capture is None:
            return
        self._ensure_capture_writer()
        self._capture_queue.put(capture)

    def _ensure_capture_writer(self) -> None:
        """Start the clip writer thread on first use (idempotent).

        The writer is deliberately NOT started in the constructor — every
        worker construction used to leak a parked thread. Stopped via the
        ``None`` sentinel from :meth:`_shutdown_capture_writer`.
        """
        with self._state_lock:
            thread = self._capture_thread
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(
                target=self._capture_writer, name="live-capture-writer", daemon=True
            )
            self._capture_thread = thread
        thread.start()

    def _release_capture(self, capture: dict) -> None:
        """Free the capture slot after encode (writer thread).

        Runs in the writer's ``finally`` so an encoding FAILURE releases the
        slot too — a failed encode must not pin capture forever.
        """
        with self._state_lock:
            if self._capture is capture:
                self._capture = None

    def _shutdown_capture_writer(self) -> None:
        """Signal writer shutdown on a worker exit path (worker thread).

        Queues the ``None`` sentinel AFTER any accepted capture — FIFO
        ordering lets an already completed capture (queued or encoding)
        finish first. An INCOMPLETE (still-collecting post-roll) capture is
        discarded. Joins the writer with a timeout — the event loop never
        joins the writer, and there is no unsafe cancellation: a hung encode
        simply outlives this join as a daemon thread.
        """
        with self._state_lock:
            if (
                self._capture is not None
                and self._capture.get("state") != "encoding"
            ):
                self._capture = None  # incomplete post-roll — discard
            thread = self._capture_thread
        if thread is None:
            return  # writer never started — nothing to stop
        self._capture_queue.put(None)
        thread.join(_CAPTURE_JOIN_TIMEOUT_S)

    def _capture_writer(self) -> None:
        """Writer-thread body: encode captures off the engine loop.

        Decoding up to ~150 pre-roll JPEGs and re-encoding an mp4 blocks for
        seconds; doing that on the worker thread froze the live view. The
        ``None`` sentinel — sent by every worker exit path AFTER its
        accepted captures — stops the thread once the accepted work is done.
        The capture slot is released in ``finally``: success or failure.
        """
        while True:
            capture = self._capture_queue.get()
            if capture is None:
                return
            try:
                self._write_capture_file(capture)
            except Exception as exc:  # noqa: BLE001 — capture must not kill anything
                self.push({"type": "status", "note": f"clip capture failed: {exc}"})
                self._prune_clips()
            finally:
                self._release_capture(capture)

    def _write_capture_file(self, capture: dict) -> None:
        """Encode one detached capture dict to an mp4 and announce it."""
        import cv2
        import numpy as np

        frames = capture.get("frames") or []
        if not frames:
            return

        name = f"clip_{int(capture['trigger_ts'])}_{capture['kind']}.mp4"
        path = _clips_dir / name
        try:
            writer = None
            for _ts, jpeg in frames:
                arr = cv2.imdecode(
                    np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if arr is None:
                    continue
                h, w = arr.shape[:2]
                w_even, h_even = w - (w % 2), h - (h % 2)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                        self._fps or 30.0, (w_even, h_even),
                    )
                    if not writer.isOpened():
                        writer = None
                        break
                if (w, h) != (w_even, h_even):
                    arr = cv2.resize(arr, (w_even, h_even))
                writer.write(arr)
            if writer is not None:
                writer.release()
        except Exception as exc:  # noqa: BLE001 — capture must not kill the engine
            self.push({"type": "status", "note": f"clip capture failed: {exc}"})
            self._prune_clips()
            return

        self.push({
            "type": "capture",
            "clip": {"name": name, "url": f"/api/clips/{name}", "kind": capture["kind"]},
        })
        self._prune_clips()

    @staticmethod
    def _prune_clips() -> None:
        """Keep only the newest ``_MAX_CLIPS`` files in the clips dir."""
        base = _clips_dir
        if base is None:
            return
        try:
            files = [p for p in base.iterdir() if p.is_file()]
            files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[_MAX_CLIPS:]:
                old.unlink(missing_ok=True)
        except OSError:
            pass

    # ── worker body ───────────────────────────────────────────────────────

    def run(self) -> None:
        """Thread entry point — never lets an exception escape.

        The ``finally`` block is the ONE choke point every exit passes
        (normal end, source failure, fatal error): it signals the clip writer
        shutdown (None sentinel) so no writer thread outlives the worker.
        """
        try:
            with self._native_condition:
                run_native = (
                    not self._stopping
                    and not self.stop_event.is_set()
                )
            if run_native:
                self._run()
        except Exception as exc:  # noqa: BLE001 — report, don't crash the process
            self.push({"type": "status", "note": f"engine error: {exc}"})
        finally:
            self._shutdown_capture_writer()

    def _run(self) -> None:
        import cv2
        import mlx.core as mx
        from PIL import Image

        from . import sam3_inference as _sam3
        from .direction_tracking import DirectionClassifier
        try:
            from mlx_vlm.models.sam3_1.generate import (
                SimpleTracker,
                _detect_with_backbone,
                _get_backbone_features,
            )
        except ImportError:  # older mlx_vlm layout
            from mlx_vlm.models.sam3.generate import (  # type: ignore[no-redef]
                SimpleTracker,
                _detect_with_backbone,
                _get_backbone_features,
            )

        def _load_sam_for_this_thread(threshold: float, resolution: int):
            """Load SAM 3.1 fresh in THIS worker thread.

            MLX binds GPU streams to the thread that loads the model:
            reusing ``sam3_inference``'s module cache from a previous worker
            dies on its first detect with "There is no Stream(gpu, 1) in
            current thread" — i.e. every second engine start in one process.
            Clearing the cache forces a clean in-thread load (~10 s) and
            lets the stale copy be collected.
            """
            _sam3._sam_model_cache.clear()
            return _sam3._ensure_sam31(threshold=threshold, resolution=resolution)

        cfg = self.cfg
        source = cfg["source"]
        threshold = float(cfg.get("threshold", 0.15))
        detect_every = max(1, int(cfg.get("detect_every", 6)))
        resolution = int(cfg.get("resolution", 1008))

        # ── open capture ──────────────────────────────────────────────────
        if source == "file":
            resolved = resolve_file_id(cfg["file_id"])
            if resolved is None:
                self.push({
                    "type": "status",
                    "note": f"could not resolve file_id {cfg['file_id']!r} in the uploads dir",
                })
                return
            cap = cv2.VideoCapture(str(resolved))
            source_desc = resolved.name
        elif source == "url":
            url = str(cfg["url"])
            # A URL can surface in status notes and telemetry — always the
            # redacted form, never raw credentials.
            source_desc = redact_url(url)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        else:
            camera = int(cfg["camera"])
            cap = cv2.VideoCapture(camera)
            source_desc = f"webcam:{camera}"

        # Honest startup notes: model load and source open are the two long
        # silent windows between `start` and the first wire frame.
        self.push({"type": "status", "note": f"opening source: {source_desc}"})

        try:
            if not cap.isOpened():
                if source == "url":
                    self.push({
                        "type": "status",
                        "note": f"stream unreachable: {source_desc}",
                    })
                else:
                    self.push({"type": "status", "note": f"cannot open source: {source_desc}"})
                return

            # ── model (private per-worker load; see _load_sam_for_this_thread)
            self.push({"type": "status", "note": "loading SAM 3.1 weights"})
            try:
                model, processor, predictor = _load_sam_for_this_thread(
                    threshold=threshold, resolution=resolution
                )
            except Exception as exc:  # noqa: BLE001 — weights missing etc.
                self.push({"type": "status", "note": f"SAM 3.1 unavailable: {exc}"})
                return

            tracker = SimpleTracker(iou_threshold=0.3, max_lost=10)
            classifier = DirectionClassifier()

            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps <= 0:
                fps = 30.0
            # File playback is paced to the video's own fps; webcam ~30fps.
            # Network streams pace themselves via their blocking reads — for
            # them the loop only applies a burst guard (see below).
            if source == "url":
                frame_interval = 0.0
                burst_min_s = 1.0 / max(fps, 1.0) / 2.0   # 2x-fps read floor
                last_read_t = time.monotonic()
            else:
                frame_interval = 1.0 / (30.0 if source == "webcam" else fps)
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # ── smart-capture per-run state ───────────────────────────────
            self._dims = (src_w, src_h)
            self._fps = 30.0 if source == "webcam" else fps
            with self._state_lock:
                pre_s = float(self._triggers.get("pre_s", TRIGGER_DEFAULTS["pre_s"]) or 0.0)
            # Pre-roll capacity: pre_s worth of frames, hard-capped. The "or
            # 30" covers pre_s = 0 (disabled pre-roll still keeps ~1s).
            self._ring = deque(maxlen=min(int(pre_s * self._fps) or 30, _MAX_RING_FRAMES))

            frame_id = 0   # wire frame counter (monotonic, even across loops)
            fi = 0         # position in the stream (resets when a file loops)
            read_fails = 0  # consecutive failed reads (url sources only)
            held_tick = 0
            latest_items: list[dict] = []
            next_frame_t = time.monotonic()
            # NO cross-pass backbone/encoder cache: every scheduled detect
            # pass recomputes image features from ITS OWN frame, so a file
            # loop or pause/resume can never dress an old scene up as a new
            # observation. ``detect_every`` is the only inference throttle.

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if source == "file" and fi > 0:
                        # Continuous playback: loop the file back to frame 0.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        fi = 0
                        classifier.reset()
                        self._dir_state.reset()
                        self._dwell.reset()
                        continue
                    if source == "url":
                        # Tolerate a bounded run of transient network
                        # hiccups; a stream that never recovers (or closes
                        # immediately) fails cleanly — no infinite retry.
                        read_fails += 1
                        if read_fails < _MAX_STREAM_READ_FAILS:
                            continue
                        self.push({
                            "type": "status",
                            "note": f"stream unreachable: {source_desc}",
                        })
                        return
                    self.push({"type": "status", "note": f"source ended: {source_desc}"})
                    return

                # Burst guard (url only): IP cameras can dump buffered
                # frames faster than real time — if reads arrive sooner
                # than 2x the target fps, slow down so the encode/send
                # loop does not spin the CPU. Live streams block inside
                # cap.read() and never trigger this.
                if source == "url":
                    read_fails = 0
                    gap = time.monotonic() - last_read_t
                    if gap < burst_min_s:
                        time.sleep(burst_min_s - gap)
                    last_read_t = time.monotonic()

                current = self.get_prompts()
                with self._tuning_lock:
                    threshold = self._threshold
                    task = self._task
                fired_events: list[dict] = []

                # ── detection: every detect_every-th frame AND frame 0 ────
                if fi % detect_every == 0:
                    if not current:
                        # Detection off — the hub-protocol pause semantics
                        # (empty prompt set). Zero model work; an empty
                        # detection set tells clients to drop stale boxes.
                        latest_items = []
                        self.push_detections([])
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        # FRESH features on EVERY scheduled pass: backbone +
                        # encoder run against this exact frame. The per-pass
                        # encoder cache dict exists only because the model
                        # helper's signature requires it — it never survives
                        # the pass (no cross-pass reuse, no separate
                        # backbone cadence; detect_every is the sole
                        # throttle).
                        inputs = processor.preprocess_image(frame_pil)
                        pixel_values = mx.array(inputs["pixel_values"])
                        backbone_features = _get_backbone_features(
                            model, pixel_values
                        )
                        result = _detect_with_backbone(
                            predictor, backbone_features, current,
                            frame_pil.size, threshold,
                            encoder_cache={},
                        )
                        latest = tracker.update(result)

                        scores = latest.scores
                        boxes = latest.boxes
                        labels = latest.labels or (current * len(scores))
                        track_ids = getattr(latest, "track_ids", None)
                        # SAM paints per-object masks in frame pixel space —
                        # MASK switch: polygons only in the segment task;
                        # detect ships fast boxes (no per-object full-frame
                        # mask tracing, much smaller payloads).
                        masks = getattr(latest, "masks", None)
                        current_targets = self.get_targets()
                        items: list[dict] = []
                        for i, (score, box, label) in enumerate(zip(scores, boxes, labels)):
                            tid = (
                                int(track_ids[i])
                                if track_ids is not None and i < len(track_ids)
                                else i
                            )
                            polygon = None
                            if task == "segment" and masks is not None and i < len(masks):
                                polygon = mask_to_polygon(masks[i], src_w, src_h)
                            item = make_item(
                                [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                                src_w,
                                src_h,
                                str(label or "object"),
                                float(score),
                                tid,
                                "unknown",
                                polygon=polygon,
                            )
                            # Box-prompted targets are ROI labels, not extra
                            # inference: box conditioning is inert in the
                            # installed mlx_vlm, so relabel objects whose
                            # center falls inside a drawn ROI (first match
                            # wins) instead of re-detecting per target.
                            if current_targets:
                                cx01 = (item["box"][0] + item["box"][2]) / 2.0
                                cy01 = (item["box"][1] + item["box"][3]) / 2.0
                                target_label = target_label_at(cx01, cy01, current_targets)
                                if target_label:
                                    item["label"] = target_label
                            items.append(item)
                        # 8-way heading per track — classify() consumes the same
                        # normalized items the client receives (box + track_id)
                        # and returns copies with direction/moved attached.
                        items = classifier.classify(items)
                        latest_items = items

                        # Snapshot for the VLM watch/ask: the full-res frame
                        # the worker already built, before JPEG downscale.
                        # This IS the evidence — the next real detect pass
                        # replaces it, a pause clears it.
                        with self._frame_lock:
                            self._latest_frame = (int(frame_id), frame_pil)
                            self._last_frame_items = list(items)

                        self.push_detections(items)

                        # Zones + deterministic triggers (line / rect /
                        # direction / dwell) — each fired event goes out as
                        # its own message.
                        fired_events = self._evaluate_triggers(items, time.time())
                        for ev in fired_events:
                            self.push({
                                "type": "event",
                                "event": {
                                    "kind": str(ev.get("kind", "")),
                                    "zone": str(ev.get("zone", "")),
                                    "direction": str(ev.get("direction", "")),
                                    "track_id": ev.get("track_id"),
                                    "ts": float(ev.get("ts", 0.0)),
                                    "frame_id": int(frame_id),
                                    "detail": str(ev.get("detail", "")),
                                },
                            })
                elif latest_items:
                    held_tick += 1
                    if held_tick % _HELD_EVERY == 0:
                        held = [dict(it, track_state="held") for it in latest_items]
                        self.push_detections(held)

                # ── clean JPEG (downscaled copy, no overlay) ──────────────
                # Quality + width are live STREAM knobs (Android parity):
                # lower quality/width cuts encode CPU and wire size with no
                # effect on detection quality.
                with self._tuning_lock:
                    send_width = self._send_width
                    jpeg_quality = self._jpeg_quality
                if src_w > send_width:
                    scale = send_width / src_w
                    small = cv2.resize(
                        frame, (send_width, max(1, int(round(src_h * scale))))
                    )
                else:
                    small = frame
                ok, buf = cv2.imencode(
                    ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                )
                if not ok:
                    self.push({"type": "status", "note": "JPEG encode failed"})
                    return

                # Ring buffer (clip pre-roll) + pending-capture bookkeeping,
                # using the SAME encoded JPEG that is streamed below. Frames
                # only grow a capture that is still "collecting" — one
                # already handed to the writer stays frozen.
                jpeg_bytes = buf.tobytes()
                now_ts = time.time()
                with self._state_lock:
                    self._ring.append((now_ts, jpeg_bytes))
                    pending = self._capture
                    if (
                        pending is not None
                        and pending.get("state") == "collecting"
                    ):
                        pending["frames"].append((now_ts, jpeg_bytes))
                        capture_due = now_ts >= float(pending["deadline"])
                    else:
                        capture_due = False
                # Arm captures only after the trigger frame is in the ring,
                # so the pre-roll snapshot includes it.
                for ev in fired_events:
                    self.request_capture(
                        str(ev.get("kind", "event")), float(ev.get("ts", now_ts))
                    )
                if capture_due:
                    # Encode off-loop: the pre-roll decode/re-encode blocks
                    # for seconds and must never run on this frame loop.
                    # The slot stays claimed until the WRITER finishes
                    # encoding, so triggers firing during a slow encode
                    # emit events but cannot allocate another capture.
                    self._dispatch_due_capture()

                # All-native telemetry — no numpy types can leak into JSON.
                telemetry = {
                    "engine": "local",
                    "source": str(source_desc),
                    "prompts": list(current),
                    "fps": round(float(fps), 2),
                    "frame_id": int(frame_id),
                    "resolution": f"{src_w}x{src_h}",
                    "threshold": float(threshold),
                    "task": str(task),
                    "jpeg_quality": int(jpeg_quality),
                    "send_width": int(send_width),
                }
                ts_ms = int(time.time() * 1000)
                self.push_frame(pack_frame(frame_id, ts_ms, jpeg_bytes, telemetry))

                # ── pace to (roughly) real time ───────────────────────────
                # File/webcam only: url streams are paced by their own
                # blocking reads plus the burst guard above.
                if source != "url":
                    next_frame_t += frame_interval
                    delay = next_frame_t - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_frame_t = time.monotonic()  # fell behind; drop debt

                frame_id += 1
                fi += 1
        finally:
            cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────────────────────

router = APIRouter()

_engine_lock = threading.Lock()
_worker: Optional[_EngineWorker] = None
_pending_zones: Optional[list[dict]] = None    # set_zones before a worker exists
_pending_triggers: Optional[dict] = None       # set_triggers before a worker exists
_pending_targets: Optional[list[dict]] = None  # add_prompt_box before a worker exists
_sinks: "set[_ClientSink]" = set()             # every attached viewer's sink


def _alive(worker: Optional[_EngineWorker]) -> bool:
    """True when a worker exists and its thread is still running."""
    return worker is not None and worker.thread is not None and worker.thread.is_alive()


async def _safe_send_json(websocket: WebSocket, message: dict) -> None:
    """Send a JSON status message, ignoring send failures on a dead socket."""
    try:
        await websocket.send_text(json.dumps(message))
    except Exception:  # noqa: BLE001 — socket already closing
        pass


async def _sender(websocket: WebSocket, sink: "_ClientSink") -> None:
    """Drain one viewer's sink queue to its socket.

    Binary frames AND ``detections`` sets travel via the sink's one-slot
    pending buffers (the queue only ever carries the ``_FRAME_SENTINEL`` /
    ``_DETECTIONS_SENTINEL`` markers), so a slow viewer can never
    accumulate a backlog of stale JPEGs or stale detection snapshots — the
    newest value wins and older ones are dropped. Other dicts (events /
    captures / status / ask_ack / answer / report_result / error) are
    json-dumped in order and never dropped.
    """
    while True:
        message = await sink.queue.get()
        try:
            if message is _FRAME_SENTINEL:
                with sink.out_lock:
                    frame = sink.pending_frame
                    sink.pending_frame = None
                if frame is not None:
                    await websocket.send_bytes(frame)
            elif message is _DETECTIONS_SENTINEL:
                with sink.out_lock:
                    items = sink.pending_detections
                    sink.pending_detections = None
                if items is not None:
                    await websocket.send_text(
                        json.dumps({"type": "detections", "items": items})
                    )
            elif isinstance(message, (bytes, bytearray)):
                # Not produced by the worker anymore; kept for robustness.
                await websocket.send_bytes(bytes(message))
            else:
                await websocket.send_text(json.dumps(message))
        except Exception:  # noqa: BLE001 — socket closed; receiver cleans up
            return


@router.websocket("/api/live/ws")
async def live_ws(websocket: WebSocket) -> None:
    """Local live engine — field-hub-compatible wire protocol over WebSocket.

    Honors ``VB_TOKEN``: when the shared token is enabled, the client must
    pass it as ``?token=`` on the connect URL (browsers cannot set headers
    on WebSocket connects); a missing or wrong token is denied at the
    handshake before ``accept()``.
    """
    global _worker, _pending_zones, _pending_triggers, _pending_targets

    # Token gate at the door, BEFORE accept: a pre-accept close becomes a
    # handshake denial. ``service`` is stdlib-only but is imported here to
    # keep the module's lazy-import discipline (web_app's HTTP middleware
    # deliberately exempts this route — the check lives in the handler).
    from . import service

    if service.token_enabled() and not service.check_token(
        websocket.query_params.get("token")
    ):
        try:
            await websocket.close(code=4401)
        except Exception:  # noqa: BLE001 — denial send failures are fine here
            pass
        return

    await websocket.accept()
    # Multi-viewer: any connected dashboard receives the running engine's
    # stream; only the SOURCE/config changes need a stop + fresh start.
    streaming = _alive(_worker)
    await _safe_send_json(
        websocket,
        {"type": "status",
         "note": ("local engine ready — engine streaming (attached as viewer)"
                  if streaming else "local engine ready")},
    )

    sink = _ClientSink(asyncio.get_running_loop())
    _sinks.add(sink)
    sender = asyncio.create_task(_sender(websocket, sink))

    try:
        while True:
            # Receive controls (JSON text) and send frames concurrently: the
            # sender task owns all frame traffic, this loop owns controls.
            try:
                raw = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:  # noqa: BLE001 — malformed frame; treat as disconnect
                break

            try:
                msg = json.loads(raw)
            except ValueError:
                await _safe_send_json(websocket, {"type": "status", "note": "invalid control — expected JSON"})
                continue

            action, payload = validate_control(msg)

            if action == "unknown":
                await _safe_send_json(websocket, {"type": "status", "note": "unknown or invalid control"})
                continue
            if action == "ignore":
                continue

            if action == "start":
                # Legacy compatibility: the key validates, but the worker never
                # sees it — features are recomputed every detection pass. The
                # note goes out after the start settles so it cannot break a
                # normal start.
                backbone_note: Optional[str] = None
                if "backbone_every" in payload:
                    del payload["backbone_every"]
                    backbone_note = (
                        "backbone_every is deprecated and ignored — features "
                        "are recomputed on every detection pass"
                    )
                # Single-instance guard: decide under the lock without awaiting,
                # then start the thread outside it (a threading.Lock must never
                # be held across an await on the event loop thread).
                created: Optional[_EngineWorker] = None
                pending_zones: Optional[list[dict]] = None
                pending_triggers: Optional[dict] = None
                pending_targets: Optional[list[dict]] = None
                admission_note: Optional[str] = None
                with _engine_lock:
                    if not _alive(_worker):
                        try:
                            admission = InferenceAdmission()
                            handle = admission.try_acquire("visionbrain-local-live")
                        except AdmissionError as exc:
                            handle = None
                            admission_note = f"inference admission unavailable: {exc}"
                        if admission_note is None and handle is None:
                            admission_note = (
                                "inference admission busy: "
                                f"{admission.describe_holder()}"
                            )
                        if handle is not None:
                            try:
                                created = _EngineWorker(payload, _sinks)
                                created._admission_handle = handle
                                created.thread = threading.Thread(
                                    target=created._thread_entry,
                                    name="live-engine-worker",
                                    daemon=True,
                                )
                            except Exception as exc:  # noqa: BLE001 — release failed start
                                handle.release()
                                created = None
                                admission_note = f"local engine failed to start: {exc}"
                            if created is not None:
                                _worker = created
                                # Consume any config staged before the worker existed.
                                pending_zones = _pending_zones
                                pending_triggers = _pending_triggers
                                pending_targets = _pending_targets
                                _pending_zones = None
                                _pending_triggers = None
                                _pending_targets = None
                if admission_note is not None:
                    await _safe_send_json(
                        websocket, {"type": "status", "note": admission_note}
                    )
                elif created is None:
                    # Multi-viewer: an engine started by another dashboard keeps
                    # streaming to this socket — nothing to do but say so.
                    await _safe_send_json(
                        websocket,
                        {"type": "status",
                         "note": "engine already running — attached as viewer (stop first to change source)"},
                    )
                else:
                    if pending_zones is not None:
                        created.set_zones(pending_zones)
                    if pending_triggers:
                        created.set_triggers(pending_triggers)
                    # Staged targets replay in add order; the label stored at
                    # add time (client-supplied or the engine-side "target N"
                    # sequence) is kept verbatim by add_target.
                    for entry in pending_targets or []:
                        created.add_target(entry["box"], entry.get("label"))
                    try:
                        created.thread.start()
                    except Exception as exc:  # noqa: BLE001 — release failed start
                        created._finish_native_owner()
                        with _engine_lock:
                            if _worker is created:
                                _worker = None
                        await _safe_send_json(
                            websocket,
                            {"type": "status", "note": f"local engine failed to start: {exc}"},
                        )
                if backbone_note is not None:
                    await _safe_send_json(
                        websocket, {"type": "status", "note": backbone_note}
                    )

            elif action == "stop":
                worker = _worker
                if not _alive(worker):
                    await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})
                else:
                    # Loop exits after the current frame; engine_stopped follows.
                    worker._begin_stopping()

            elif action == "shutdown":
                worker = _worker
                if worker is None:
                    await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})
                else:
                    worker._begin_stopping()
                    if worker.thread is not None and worker.thread.is_alive():
                        await asyncio.to_thread(
                            worker.thread.join, _ENGINE_JOIN_TIMEOUT_S
                        )
                    freed = False
                    cache_cleared = False
                    with _engine_lock:
                        if _worker is worker and not _alive(worker):
                            try:
                                from . import sam3_inference

                                sam3_inference._sam_model_cache.clear()
                                cache_cleared = True
                            except Exception:  # noqa: BLE001 — best-effort model free
                                pass
                            _worker = None
                            freed = True
                    if freed and cache_cleared:
                        note = "model freed"
                    elif freed:
                        note = "engine stopped — model cache remains resident"
                    elif _alive(worker):
                        note = "engine draining — admission and model retained"
                    else:
                        note = "engine ownership changed — model retained"
                    await _safe_send_json(websocket, {"type": "status", "note": note})

            elif action == "set_prompts":
                worker = _worker
                if _alive(worker):
                    worker.set_prompts(payload["prompts"])
                    note = (
                        "prompts updated"
                        if payload["prompts"]
                        else "detection paused (no prompts)"
                    )
                    await _safe_send_json(websocket, {"type": "status", "note": note})
                else:
                    await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})

            elif action == "set_threshold":
                worker = _worker
                if _alive(worker):
                    worker.set_threshold(payload["threshold"])
                    await _safe_send_json(
                        websocket,
                        {"type": "status", "note": f"threshold {payload['threshold']:g}"},
                    )
                else:
                    await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})

            elif action == "set_stream":
                worker = _worker
                if _alive(worker):
                    worker.set_stream(payload.get("jpeg_quality"), payload.get("send_width"))
                    await _safe_send_json(websocket, {"type": "status", "note": "stream retuned"})
                else:
                    await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})

            elif action == "set_task":
                worker = _worker
                if _alive(worker):
                    worker.set_task(payload["task"])
                    await _safe_send_json(
                        websocket,
                        {"type": "status", "note": f"task {payload['task']}"},
                    )
                else:
                    await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})

            elif action == "set_engine":
                # The local engine runs SAM only; answer the hub-protocol chips
                # honestly instead of silently no-oping.
                extra = [k for k in ("falcon", "lfm") if payload.get(k)]
                note = "local engine · SAM only"
                if extra:
                    note += " — not available locally: " + ", ".join(extra)
                await _safe_send_json(websocket, {"type": "status", "note": note})

            elif action == "set_vlm":
                from . import vlm_registry

                try:
                    vlm_registry.set_model(payload["model"])
                    await _safe_send_json(
                        websocket,
                        {"type": "status", "note": f"vlm · {payload['model']}"},
                    )
                except ValueError:
                    await _safe_send_json(
                        websocket,
                        {"type": "status",
                         "note": f"unknown vlm {payload['model']!r} (gemma | lfm | lfm3b)"},
                    )

            elif action == "ask":
                worker = _worker
                if not _alive(worker):
                    # Pinned error shape (ask/report path only): rejection (a).
                    await _safe_send_json(
                        websocket,
                        {"type": "error", "error": "no engine running — start the engine first"},
                    )
                else:
                    # Grounding BEFORE the shared slot: capture ONE consistent
                    # server-owned evidence snapshot (frame + records under a
                    # single _frame_lock acquisition, plus prompts) and reject
                    # unarmed/unobserved requests without any VLM inference.
                    snapshot = worker.evidence_snapshot()
                    if snapshot is None:
                        await _safe_send_json(
                            websocket,
                            {"type": "error",
                             "error": "no current observation — start the engine and arm prompts"},
                        )
                    elif not snapshot["prompts"]:
                        await _safe_send_json(
                            websocket,
                            {"type": "error",
                             "error": "prompts are paused — arm prompts to ask"},
                        )
                    elif not _claim_ask_slot():
                        await _safe_send_json(
                            websocket,
                            {"type": "error", "error": "ask/report already running"},
                        )
                    elif not worker._register_native_job():
                        _release_ask_slot()
                        await _safe_send_json(
                            websocket,
                            {"type": "error", "error": "engine is stopping — ask refused"},
                        )
                    else:
                        handed_off = False
                        try:
                            await _safe_send_json(
                                websocket,
                                {"type": "ask_ack", "model": _vlm_label()},
                            )
                            threading.Thread(
                                target=_ask_once,
                                args=(worker, payload["question"], snapshot),
                                name="live-ask", daemon=True,
                            ).start()
                            handed_off = True
                        except BaseException as exc:  # noqa: BLE001 — include ack cancellation
                            if handed_off:
                                raise
                            try:
                                _release_ask_slot()
                            finally:
                                worker._finish_native_job()
                            if isinstance(exc, Exception):
                                worker.push({"type": "error", "error": f"ask failed: {exc}"})
                            else:
                                raise

            elif action == "report":
                worker = _worker
                if not _alive(worker):
                    # Pinned error shape (ask/report path only): rejection (a).
                    await _safe_send_json(
                        websocket,
                        {"type": "error", "error": "no engine running — start the engine first"},
                    )
                else:
                    # Same evidence + grounding rule as ask: snapshot at
                    # dispatch, reject without inference, then claim the slot.
                    # Counts come from the snapshot's records SERVER-SIDE; the
                    # browser summary is never forwarded to the model.
                    snapshot = worker.evidence_snapshot()
                    if snapshot is None:
                        await _safe_send_json(
                            websocket,
                            {"type": "error",
                             "error": "no current observation — start the engine and arm prompts"},
                        )
                    elif not snapshot["prompts"]:
                        await _safe_send_json(
                            websocket,
                            {"type": "error",
                             "error": "prompts are paused — arm prompts to report"},
                        )
                    elif not _claim_ask_slot():
                        await _safe_send_json(
                            websocket,
                            {"type": "error", "error": "ask/report already running"},
                        )
                    elif not worker._register_native_job():
                        _release_ask_slot()
                        await _safe_send_json(
                            websocket,
                            {"type": "error", "error": "engine is stopping — report refused"},
                        )
                    else:
                        handed_off = False
                        try:
                            threading.Thread(
                                target=_report_once,
                                args=(worker, snapshot, payload["report_type"]),
                                name="live-report", daemon=True,
                            ).start()
                            handed_off = True
                        except BaseException as exc:  # noqa: BLE001 — release failed handoff
                            if handed_off:
                                raise
                            try:
                                _release_ask_slot()
                            finally:
                                worker._finish_native_job()
                            if isinstance(exc, Exception):
                                worker.push({"type": "error", "error": f"report failed: {exc}"})
                            else:
                                raise

            elif action == "set_zones":
                zones = payload["zones"]
                worker = _worker
                if _alive(worker):
                    # Worker rebuilds counters under its state lock on the next
                    # detect frame (which also resets line-cross totals).
                    worker.set_zones(zones)
                else:
                    # No engine yet — hold pending, applied on the next start.
                    with _engine_lock:
                        _pending_zones = zones
                await _safe_send_json(
                    websocket, {"type": "status", "note": f"zones set ({len(zones)})"}
                )

            elif action == "add_prompt_box":
                worker = _worker
                if _alive(worker):
                    number = worker.add_target(payload["box"], payload.get("label"))
                    if number is None:
                        await _safe_send_json(
                            websocket,
                            {"type": "status", "note": f"target limit reached ({_MAX_TARGETS})"},
                        )
                    else:
                        await _safe_send_json(
                            websocket, {"type": "status", "note": f"target added ({number})"}
                        )
                else:
                    # No engine yet — stage the target, applied on the next
                    # start. Mutate under the lock, send OUTSIDE it (a
                    # threading.Lock must never be held across an await on the
                    # event loop thread — it is not coroutine-aware; a suspended
                    # send would block every other handler on acquire).
                    staged_ok: list[dict] | None = None
                    staged_full = False
                    with _engine_lock:
                        staged = list(_pending_targets or [])
                        if len(staged) >= _MAX_TARGETS:
                            staged_full = True
                        else:
                            staged.append(payload)
                            _pending_targets = staged
                            staged_ok = staged
                    if staged_full:
                        await _safe_send_json(
                            websocket,
                            {"type": "status", "note": f"target limit reached ({_MAX_TARGETS})"},
                        )
                    else:
                        await _safe_send_json(
                            websocket,
                            {"type": "status",
                             "note": f"target added ({len(staged_ok or [])}) (starts with the engine)"},
                        )

            elif action == "remove_targets":
                worker = _worker
                if _alive(worker):
                    worker.clear_targets()
                else:
                    with _engine_lock:
                        _pending_targets = None
                await _safe_send_json(websocket, {"type": "status", "note": "targets cleared"})

            elif action == "set_triggers":
                worker = _worker
                if _alive(worker):
                    worker.set_triggers(payload)
                else:
                    with _engine_lock:
                        merged = dict(_pending_triggers or TRIGGER_DEFAULTS)
                        merged.update(payload)
                        _pending_triggers = merged
                await _safe_send_json(websocket, {"type": "status", "note": "triggers set"})

            elif action == "set_watch":
                enabled = await _apply_watch(payload, _worker)
                await _safe_send_json(
                    websocket,
                    {"type": "status", "note": "watch on" if enabled else "watch off"},
                )

    finally:
        _sinks.discard(sink)
        worker = _worker
        try:
            if _alive(worker) and not _sinks:
                worker._begin_stopping()
                if worker.thread is not None:
                    await asyncio.to_thread(
                        worker.thread.join, _ENGINE_JOIN_TIMEOUT_S
                    )
        finally:
            sender.cancel()
            with _engine_lock:
                if _worker is worker and worker is not None and not _alive(worker):
                    _worker = None
