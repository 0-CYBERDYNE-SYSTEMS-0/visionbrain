"""Local live engine — a WebSocket endpoint that streams SAM 3.1 detections.

Purpose
-------
The live tab normally connects as a WebSocket *client* to an external field
hub. This module lets the VisionBrain app itself play that server role: it
runs SAM 3.1 detection over an uploaded video file or a local webcam and
streams the SAME wire protocol, so the existing browser client works
unchanged. Frames are sent CLEAN (no overlay) — the client draws boxes from
the detection items itself.

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
        "color_id": int, "direction": str, "track_state"?: str}]}
    {"type": "engine_stopped"}

Control protocol (inbound JSON text, one action per message)
------------------------------------------------------------
    {"action": "start", "source": "file", "file_id": str,
     "prompts": [str, ...]}              -> run on an uploaded file
    {"action": "start", "source": "webcam", "camera": int >= 0,
     "prompts": [str, ...]}              -> run on a local camera
    {"action": "set_prompts", "prompts": [str, ...]} -> swap prompts live
    {"action": "stop"}                   -> exit after the current frame
    {"action": "shutdown"}               -> stop + free the SAM 3.1 model

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
* Heavy imports (cv2, PIL, mlx, mlx_vlm, sam3_inference) happen INSIDE the
  worker function, so CI — with no mlx and no cached weights — can import
  this module and unit-test the pure helpers.
* ``configure()`` pins an uploads directory; file sources resolve
  ``file_id`` to the SINGLE glob match of ``f"{file_id}*"`` inside it
  (mirrors ``web_app._find_upload``). Raw paths are never accepted: any path
  separator or ``..`` in a file_id is rejected up front, and the match must
  resolve inside the configured directory — path traversal is impossible.
"""

from __future__ import annotations

import asyncio
import json
import struct
import threading
from pathlib import Path
from typing import Any, Optional, Sequence

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

__all__ = [
    "router",
    "configure",
    "resolve_file_id",
    "pack_frame",
    "make_item",
    "validate_control",
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
) -> dict:
    """Build one detection item in the client protocol.

    The pixel box is normalized to ``[x1, y1, x2, y2]`` in 0-1 (clamped),
    score is rounded to 3 decimals, and ``color_id`` mirrors ``track_id`` so
    the client palette is stable per track.
    """
    w = max(1.0, float(width))
    h = max(1.0, float(height))

    def _norm(value: float, span: float) -> float:
        return round(max(0.0, min(1.0, float(value) / span)), 4)

    return {
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
    ``"set_prompts"``, ``"stop"``, ``"shutdown"``, ``"ignore"`` (protocol
    hello) — or ``"unknown"`` with an empty payload for anything invalid.
    Control key is ``type`` (hub-protocol style) or ``action`` as an alias.
    Never raises.
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
        if source not in ("file", "webcam"):
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
        else:
            camera = msg.get("camera")
            if isinstance(camera, bool) or not isinstance(camera, int) or camera < 0:
                return ("unknown", {})
            payload["camera"] = camera

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

    if action in ("stop", "shutdown"):
        return (action, {})

    return ("unknown", {})


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (uploads dir) + file_id resolution with traversal guard
# ──────────────────────────────────────────────────────────────────────────────

_uploads_dir: Optional[Path] = None


def configure(uploads_dir: Path) -> None:
    """Pin the uploads directory used to resolve ``file_id`` sources.

    Must be called (typically from web_app startup with its UPLOADS dir)
    before file-based starts will resolve; webcam starts need no config.
    """
    global _uploads_dir
    _uploads_dir = Path(uploads_dir)


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
# Worker — one running engine (thread + outbound queue + live prompts)
# ──────────────────────────────────────────────────────────────────────────────

_MAX_SEND_WIDTH = 1280
_JPEG_QUALITY = 70
_HELD_EVERY = 5  # resend held boxes every Nth non-detect frame


class _EngineWorker:
    """A single running engine: worker thread state plus tunables.

    Holds the stop event, the outbound queue target (event loop captured at
    start), and the current prompts under a small lock so ``set_prompts``
    can retarget detection mid-run from the WS thread.
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

    def get_prompts(self) -> list[str]:
        """Return a copy of the current prompts (thread-safe)."""
        with self._prompts_lock:
            return list(self._prompts)

    def set_prompts(self, prompts: list[str]) -> None:
        """Replace the active prompts (thread-safe; picked up next frame)."""
        with self._prompts_lock:
            self._prompts = list(prompts)

    def push(self, message: Any) -> None:
        """Queue an outbound message (bytes or dict) from the worker thread."""
        try:
            self.loop.call_soon_threadsafe(self.outbound.put_nowait, message)
        except RuntimeError:
            # Event loop is gone (client disconnected) — nothing to send to.
            self.stop_event.set()

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
        import time

        import cv2
        import mlx.core as mx
        from PIL import Image

        from .direction_tracking import DirectionClassifier
        from .sam3_inference import _ensure_sam31
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
        else:
            camera = int(cfg["camera"])
            cap = cv2.VideoCapture(camera)
            source_desc = f"webcam:{camera}"

        try:
            if not cap.isOpened():
                self.push({"type": "status", "note": f"cannot open source: {source_desc}"})
                return

            # ── model (lazy load; cached inside sam3_inference) ───────────
            try:
                model, processor, predictor = _ensure_sam31(
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
            frame_interval = 1.0 / (30.0 if source == "webcam" else fps)
            src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            frame_id = 0   # wire frame counter (monotonic, even across loops)
            fi = 0         # position in the stream (resets when a file loops)
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
                        continue
                    self.push({"type": "status", "note": f"source ended: {source_desc}"})
                    return

                current = self.get_prompts()

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
                    latest = tracker.update(result)

                    scores = latest.scores
                    boxes = latest.boxes
                    labels = latest.labels or (current * len(scores))
                    track_ids = getattr(latest, "track_ids", None)
                    items: list[dict] = []
                    for i, (score, box, label) in enumerate(zip(scores, boxes, labels)):
                        tid = (
                            int(track_ids[i])
                            if track_ids is not None and i < len(track_ids)
                            else i
                        )
                        items.append(make_item(
                            [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                            src_w,
                            src_h,
                            str(label or "object"),
                            float(score),
                            tid,
                            "unknown",
                        ))
                    # 8-way heading per track — classify() consumes the same
                    # normalized items the client receives (box + track_id)
                    # and returns copies with direction/moved attached.
                    items = classifier.classify(items)
                    latest_items = items
                    self.push({"type": "detections", "items": items})
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
    """Local live engine — field-hub-compatible wire protocol over WebSocket."""
    global _worker

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
            continue

        if action == "start":
            # Single-instance guard: decide under the lock without awaiting,
            # then start the thread outside it (a threading.Lock must never
            # be held across an await on the event loop thread).
            created: Optional[_EngineWorker] = None
            with _engine_lock:
                if not _alive(_worker):
                    created = _EngineWorker(payload, loop, outbound)
                    created.thread = threading.Thread(
                        target=created.run, name="live-engine-worker", daemon=True
                    )
                    _worker = created
            if created is None:
                await _safe_send_json(websocket, {"type": "status", "note": "engine busy — stop first"})
            else:
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

    # Disconnect (or fatal receive error): stop the worker, drain the sender.
    worker = _worker
    if _alive(worker):
        worker.stop_event.set()
        await asyncio.to_thread(worker.thread.join, 10.0)
    sender.cancel()
    with _engine_lock:
        if _worker is not None and not _alive(_worker):
            _worker = None
