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
    {"type": "set_prompts", "prompts": [str, ...]} -> swap prompts live
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
"detect_every" (int >= 1, default 6), "resolution" (int, default 1008).
Anything that fails validation returns ("unknown", {}) and earns a status
note — never an exception.

Design notes
------------
* ONE engine worker per process: a module-level handle guarded by
  ``_engine_lock``. A second "start" while a worker is alive is rejected with
  ``"engine busy — stop first"``.
* The worker runs on a daemon ``threading.Thread`` (MLX inference blocks);
  it pushes outbound messages onto an ``asyncio.Queue`` via
  ``loop.call_soon_threadsafe`` (the running loop is captured at start). The
  WS handler drains that queue with a sender task, so receiving controls and
  sending frames run concurrently.
* The VLM event watch runs on its OWN daemon thread (started/stopped by
  ``set_watch``): it sleeps ``interval_s``, snapshots the worker's latest
  full-resolution frame and asks the configured local VLM whether the
  condition holds. Import of ``vlm_registry`` happens inside the watcher
  thread so this module stays CI-importable; watch failures produce status
  notes (throttled to one per 30s) and a model that cannot load disables the
  watch with a single note.
* Clip capture: the worker keeps a ring buffer of recent encoded JPEGs; when
  a trigger fires it snapshots the pre-roll, accumulates frames for
  ``post_s``, then writes an mp4 into the clips directory (pruned to the 50
  newest files). Only one capture is pending at a time — further triggers
  still emit events but do not capture.
* Box targets (``add_prompt_box``) pair the documented ``predict(boxes=…)``
  API with ROI-containment labeling: the installed mlx_vlm build plumbs the
  ``boxes`` kwarg but never applies box conditioning (its geometry encoder
  is never called), so targets track text-prompt detections whose centers
  fall inside the drawn ROI. Overlapping ROIs can double-count objects in
  the overlap — see ``_detect_box_targets``.
* Heavy imports (cv2, PIL, numpy, mlx, mlx_vlm, supervision, sam3_inference)
  happen INSIDE the worker / watcher bodies, so CI — with no mlx and no
  cached weights — can import this module and unit-test the pure helpers.
* ``configure()`` pins an uploads directory and a clips directory; file
  sources resolve ``file_id`` to the SINGLE glob match of ``f"{file_id}*"``
  inside it (mirrors ``web_app._find_upload``). Raw paths are never accepted:
  any path separator or ``..`` in a file_id is rejected up front, and the
  match must resolve inside the configured directory — path traversal is
  impossible.
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
import re
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional, Sequence

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .detection_core import mask_to_polygon

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
    return header + bytes(jpeg) + struct.pack(">I", len(telem_bytes)) + telem_bytes


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


def _num01_or_more(value: Any, minimum: float) -> bool:
    """True when value is a real number >= minimum (bools rejected)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) >= minimum


def _valid_prompts(value: Any) -> Optional[list[str]]:
    """Return the prompts list when valid, else None.

    Valid means a non-empty list of non-empty strings.
    """
    if not isinstance(value, list) or not value:
        return None
    for prompt in value:
        if not isinstance(prompt, str) or not prompt.strip():
            return None
    return list(value)


def validate_control(msg: Any) -> tuple[str, dict]:
    """Validate one inbound control message.

    Returns ``(action, payload)`` where action is one of ``"start"``,
    ``"set_prompts"``, ``"set_zones"``, ``"add_prompt_box"``,
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


def _watch_loop(stop_event: threading.Event) -> None:
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

        worker = _worker
        if not _alive(worker) or worker.stop_event.is_set():
            continue
        condition = str(cfg.get("condition") or "").strip()
        if not condition:
            continue
        with _watch_lock:
            if _watch_busy:
                continue
            _watch_busy = True
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
            with _watch_lock:
                _watch_busy = False


async def _apply_watch(payload: dict) -> bool:
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
        with _watch_lock:
            if _watch_thread is None or not _watch_thread.is_alive():
                _watch_stop = threading.Event()
                _watch_thread = threading.Thread(
                    target=_watch_loop, args=(_watch_stop,),
                    name="live-watch", daemon=True,
                )
                _watch_thread.start()
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

_MAX_SEND_WIDTH = 1280
_JPEG_QUALITY = 70
_HELD_EVERY = 5  # resend held boxes every Nth non-detect frame
_MAX_RING_FRAMES = 150
_MAX_CLIPS = 50
_MAX_STREAM_READ_FAILS = 40  # consecutive url read failures before giving up
_BOX_NOTE_THROTTLE_S = 30.0  # min gap between box-target error status notes


class _EngineWorker:
    """A single running engine: worker thread state plus tunables.

    Holds the stop event, the outbound queue target (event loop captured at
    start), and the current prompts under a small lock so ``set_prompts``
    can retarget detection mid-run from the WS thread. Also owns the shared
    smart-capture state (zones, box-prompted targets, triggers, ring buffer,
    pending capture, latest full frame) guarded by ``_state_lock`` /
    ``_frame_lock``.
    """

    def __init__(
        self,
        cfg: dict,
        loop: asyncio.AbstractEventLoop,
        outbound: "asyncio.Queue[Any]",
    ) -> None:
        self.cfg = cfg
        self.loop = loop
        self.outbound = outbound
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
        self._box_note_ts = 0.0          # monotonic ts of last box-target note
        self._triggers: dict[str, Any] = dict(TRIGGER_DEFAULTS)
        self._ring: deque = deque()
        self._capture: Optional[dict] = None
        self._fps = 30.0
        self._dims: tuple[int, int] = (0, 0)
        self._dir_state = DirectionTriggerState()
        self._dwell = DwellTracker(0.0)

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[tuple[int, Any]] = None

    def get_prompts(self) -> list[str]:
        """Return a copy of the current prompts (thread-safe)."""
        with self._prompts_lock:
            return list(self._prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        """Replace the active prompts (thread-safe; picked up next frame)."""
        with self._prompts_lock:
            self._prompts = list(prompts)

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

    def _note_box_target_error(self, exc: Exception) -> None:
        """Push a box-target failure note, throttled to one per 30s."""
        now = time.monotonic()
        if now - self._box_note_ts < _BOX_NOTE_THROTTLE_S:
            return
        self._box_note_ts = now
        self.push({"type": "status", "note": f"box target error: {exc}"})

    def _detect_box_targets(
        self,
        predictor: Any,
        frame_pil: Any,
        prompts: list[str],
        targets: list[dict],
        result: Any,
        threshold: float,
        src_w: int,
        src_h: int,
    ) -> Any:
        """Run box-guided detection per target ROI and merge into ``result``.

        Each target re-detects every detect frame through the documented
        box-guided API ``Sam3Predictor.predict(image, text_prompt, boxes,
        score_threshold)`` — ``boxes`` is an ``(N, 4)`` float ndarray of
        normalized 0-1 xyxy coordinates (the space the predictor's own
        postprocess scales out of), and the returned ``DetectionResult``
        carries pixel xyxy boxes, masks and scores without labels.

        IMPORTANT CAVEAT: the installed mlx_vlm build plumbs the ``boxes``
        kwarg through ``predict`` but never applies box conditioning (the
        geometry encoder is instantiated but never called), so the geometry
        itself is inert. The text-prompt detections are therefore assigned
        to the first target ROI containing their center (persistent ROI
        semantics) and labeled with that target's label ("target N" when
        the client omitted one). Overlapping ROIs can double-count an
        object sitting in the overlap. A per-target failure pushes a
        throttled status note and skips that target for this frame only —
        the worker keeps running.
        """
        import numpy as np

        try:
            from mlx_vlm.models.sam3_1.generate import DetectionResult, nms
        except ImportError:  # older mlx_vlm layout
            from mlx_vlm.models.sam3.generate import (  # type: ignore[no-redef]
                DetectionResult,
                nms,
            )

        text_prompt = ", ".join(prompts) if prompts else "object"
        box_parts = [np.asarray(result.boxes)]
        score_parts = [np.asarray(result.scores)]
        mask_parts = [np.asarray(result.masks)]
        label_parts = [list(result.labels or [])]
        added = 0

        for target in targets:
            try:
                roi = np.array([target["box"]], dtype=np.float32)
                sub = predictor.predict(
                    frame_pil,
                    text_prompt=text_prompt,
                    boxes=roi,
                    score_threshold=threshold,
                )
                if sub is not None and len(getattr(sub, "scores", [])) > 0:
                    sub = nms(sub)
            except Exception as exc:  # noqa: BLE001 — never kill the worker
                self._note_box_target_error(exc)
                continue
            if sub is None or len(sub.scores) == 0:
                continue

            kept_boxes, kept_scores, kept_masks, kept_labels = [], [], [], []
            for i in range(len(sub.scores)):
                bx = sub.boxes[i]
                cx01 = (float(bx[0]) + float(bx[2])) / (2.0 * max(1.0, float(src_w)))
                cy01 = (float(bx[1]) + float(bx[3])) / (2.0 * max(1.0, float(src_h)))
                label = target_label_at(cx01, cy01, targets)
                if label is None:
                    continue
                kept_boxes.append(np.asarray(bx, dtype=np.float32))
                kept_scores.append(float(sub.scores[i]))
                kept_masks.append(np.asarray(sub.masks[i]))
                kept_labels.append(label)
            if not kept_boxes:
                continue
            box_parts.append(np.stack(kept_boxes))
            score_parts.append(np.asarray(kept_scores, dtype=np.float32))
            mask_parts.append(np.stack(kept_masks))
            label_parts.append(kept_labels)
            added += len(kept_boxes)

        if not added:
            return result
        return DetectionResult(
            boxes=np.concatenate(box_parts),
            masks=np.concatenate(mask_parts),
            scores=np.concatenate(score_parts),
            labels=[label for part in label_parts for label in part],
        )

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
        """Arm one pending clip capture; False when disabled/already pending.

        Snapshots the current ring buffer as pre-roll; the worker accumulates
        frames until ``trigger_ts + post_s`` and then writes the file.
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
            }
            return True

    def push(self, message: Any) -> None:
        """Queue an outbound message (bytes or dict) from the worker thread."""
        try:
            self.loop.call_soon_threadsafe(self.outbound.put_nowait, message)
        except RuntimeError:
            # Event loop is gone (client disconnected) — nothing to send to.
            self.stop_event.set()

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

    def _write_capture(self) -> None:
        """Write the pending capture's buffered JPEGs to an mp4 and announce it.

        Runs on the worker thread (brief blocking is fine); only one capture
        is pending at a time. Prunes the clips dir to the newest files after.
        """
        import cv2
        import numpy as np

        with self._state_lock:
            capture = self._capture
            self._capture = None
        if not capture or _clips_dir is None:
            return
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
        """Thread entry point — never lets an exception escape."""
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 — report, don't crash the process
            self.push({"type": "status", "note": f"engine error: {exc}"})
        finally:
            self.push({"type": "engine_stopped"})

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
                fired_events: list[dict] = []

                # ── detection: every detect_every-th frame AND frame 0 ────
                if fi % detect_every == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    inputs = processor.preprocess_image(frame_pil)
                    pixel_values = mx.array(inputs["pixel_values"])
                    # Backbone recompute per detect is acceptable for the
                    # first cut — no memory-bank/propagation machinery here.
                    backbone = _get_backbone_features(model, pixel_values)
                    result = _detect_with_backbone(
                        predictor, backbone, current, frame_pil.size, threshold
                    )
                    # Box-prompted targets re-detect on the SAME frame and
                    # merge into the SAME result, so they pick up track IDs
                    # and flow through zones/triggers like text detections.
                    current_targets = self.get_targets()
                    if current_targets:
                        result = self._detect_box_targets(
                            predictor, frame_pil, current, current_targets,
                            result, threshold, src_w, src_h,
                        )
                    latest = tracker.update(result)

                    scores = latest.scores
                    boxes = latest.boxes
                    labels = latest.labels or (current * len(scores))
                    track_ids = getattr(latest, "track_ids", None)
                    # SAM paints per-object masks in frame pixel space — the
                    # whole point of tracking this model. Extract each mask's
                    # outline so the dashboard can paint the object, not a
                    # rectangle around it.
                    masks = getattr(latest, "masks", None)
                    items: list[dict] = []
                    for i, (score, box, label) in enumerate(zip(scores, boxes, labels)):
                        tid = (
                            int(track_ids[i])
                            if track_ids is not None and i < len(track_ids)
                            else i
                        )
                        polygon = None
                        if masks is not None and i < len(masks):
                            polygon = mask_to_polygon(masks[i], src_w, src_h)
                        items.append(make_item(
                            [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            src_w,
                            src_h,
                            str(label or "object"),
                            float(score),
                            tid,
                            "unknown",
                            polygon=polygon,
                        ))
                    # 8-way heading per track — classify() consumes the same
                    # normalized items the client receives (box + track_id)
                    # and returns copies with direction/moved attached.
                    items = classifier.classify(items)
                    latest_items = items

                    # Snapshot for the VLM watch: the full-res frame the
                    # worker already built, before JPEG downscale.
                    with self._frame_lock:
                        self._latest_frame = (int(frame_id), frame_pil)

                    self.push({"type": "detections", "items": items})

                    # Zones + deterministic triggers (line / rect / direction
                    # / dwell) — each fired event goes out as its own message.
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
                        self.push({"type": "detections", "items": held})

                # ── clean JPEG (downscaled copy, no overlay) ──────────────
                if src_w > _MAX_SEND_WIDTH:
                    scale = _MAX_SEND_WIDTH / src_w
                    small = cv2.resize(
                        frame, (_MAX_SEND_WIDTH, max(1, int(round(src_h * scale))))
                    )
                else:
                    small = frame
                ok, buf = cv2.imencode(
                    ".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
                )
                if not ok:
                    self.push({"type": "status", "note": "JPEG encode failed"})
                    return

                # Ring buffer (clip pre-roll) + pending-capture bookkeeping,
                # using the SAME encoded JPEG that is streamed below.
                jpeg_bytes = buf.tobytes()
                now_ts = time.time()
                with self._state_lock:
                    self._ring.append((now_ts, jpeg_bytes))
                    pending = self._capture
                    if pending is not None:
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
                    self._write_capture()

                # All-native telemetry — no numpy types can leak into JSON.
                telemetry = {
                    "engine": "local",
                    "source": str(source_desc),
                    "prompts": list(current),
                    "fps": round(float(fps), 2),
                    "frame_id": int(frame_id),
                    "resolution": f"{src_w}x{src_h}",
                    "threshold": float(threshold),
                }
                ts_ms = int(time.time() * 1000)
                self.push(pack_frame(frame_id, ts_ms, buf.tobytes(), telemetry))

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


def _alive(worker: Optional[_EngineWorker]) -> bool:
    """True when a worker exists and its thread is still running."""
    return worker is not None and worker.thread is not None and worker.thread.is_alive()


async def _safe_send_json(websocket: WebSocket, message: dict) -> None:
    """Send a JSON status message, ignoring send failures on a dead socket."""
    try:
        await websocket.send_text(json.dumps(message))
    except Exception:  # noqa: BLE001 — socket already closing
        pass


async def _sender(websocket: WebSocket, outbound: "asyncio.Queue[Any]") -> None:
    """Drain the worker's outbound queue to the socket.

    Binary bytes (packed frames) go out as-is; dicts are json-dumped.
    """
    while True:
        message = await outbound.get()
        try:
            if isinstance(message, (bytes, bytearray)):
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
    await _safe_send_json(websocket, {"type": "status", "note": "local engine ready"})

    outbound: "asyncio.Queue[Any]" = asyncio.Queue()
    loop = asyncio.get_running_loop()
    sender = asyncio.create_task(_sender(websocket, outbound))

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
            # Single-instance guard: decide under the lock without awaiting,
            # then start the thread outside it (a threading.Lock must never
            # be held across an await on the event loop thread).
            created: Optional[_EngineWorker] = None
            pending_zones: Optional[list[dict]] = None
            pending_triggers: Optional[dict] = None
            pending_targets: Optional[list[dict]] = None
            with _engine_lock:
                if not _alive(_worker):
                    created = _EngineWorker(payload, loop, outbound)
                    created.thread = threading.Thread(
                        target=created.run, name="live-engine-worker", daemon=True
                    )
                    _worker = created
                    # Consume any config staged before the worker existed.
                    pending_zones = _pending_zones
                    pending_triggers = _pending_triggers
                    pending_targets = _pending_targets
                    _pending_zones = None
                    _pending_triggers = None
                    _pending_targets = None
            if created is None:
                await _safe_send_json(websocket, {"type": "status", "note": "engine busy — stop first"})
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
                created.thread.start()

        elif action == "stop":
            worker = _worker
            if not _alive(worker):
                await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})
            else:
                # Loop exits after the current frame; engine_stopped follows.
                worker.stop_event.set()

        elif action == "shutdown":
            worker = _worker
            if not _alive(worker):
                await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})
            else:
                worker.stop_event.set()
                await asyncio.to_thread(worker.thread.join, 10.0)
                try:
                    from . import sam3_inference

                    sam3_inference._sam_model_cache.clear()
                except Exception:  # noqa: BLE001 — best-effort model free
                    pass
                with _engine_lock:
                    if _worker is worker:
                        _worker = None
                await _safe_send_json(websocket, {"type": "status", "note": "model freed"})

        elif action == "set_prompts":
            worker = _worker
            if _alive(worker):
                worker.set_prompts(payload["prompts"])
                await _safe_send_json(websocket, {"type": "status", "note": "prompts updated"})
            else:
                await _safe_send_json(websocket, {"type": "status", "note": "no engine running"})

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
                # No engine yet — stage the target, applied on the next start.
                with _engine_lock:
                    staged = list(_pending_targets or [])
                    if len(staged) >= _MAX_TARGETS:
                        await _safe_send_json(
                            websocket,
                            {"type": "status", "note": f"target limit reached ({_MAX_TARGETS})"},
                        )
                    else:
                        staged.append(payload)
                        _pending_targets = staged
                        await _safe_send_json(
                            websocket,
                            {"type": "status",
                             "note": f"target added ({len(staged)}) (starts with the engine)"},
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
            enabled = await _apply_watch(payload)
            await _safe_send_json(
                websocket,
                {"type": "status", "note": "watch on" if enabled else "watch off"},
            )

    # Disconnect (or fatal receive error): stop the worker, drain the sender.
    worker = _worker
    if _alive(worker):
        worker.stop_event.set()
        await asyncio.to_thread(worker.thread.join, 10.0)
    sender.cancel()
    with _engine_lock:
        if _worker is not None and not _alive(_worker):
            _worker = None
