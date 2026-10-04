"""Live SAM 3.1 tracking as an importable, per-frame primitive.

The stateful half of live tracking: call :meth:`LiveSamTracker.step` once per
frame with the current image and get stable-identity detections back. This is
the field-hardened implementation (lifted from the bridge after weeks of
drone duty), now canonical for every surface — the Ground Control console can
drive the exact same tracker the cockpit HUD uses.

Design choices (deliberately lighter than ``track_video_realtime``):

* The ViT backbone is recomputed only every ``backbone_every`` detect passes
  (or sooner once it is ``max_backbone_age_ms`` old); ``_detect_with_backbone``
  reuses the cached features in between — the 2-4x speed win,
  resolution-independent. Geometry from a cached backbone describes the frame
  that backbone saw, so items carry that frame's id/timestamp and an honest
  ``stale_ms``.
* Between detect frames the last detection set is re-emitted with the same
  IDs ("held"), stamped with ``track_state="predicted"`` and honest
  ``stale_ms`` measured on the source's own clock, so consumers can tell lag
  from a fresh measurement.
* SAM's memory-bank *propagation* path is not wired: it only works at native
  1008px and is coupled to the video ingest loop.

The model loader and the two detection/backbone helpers are injectable so the
module imports and unit-tests cleanly without mlx_vlm or weights.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Optional

log = logging.getLogger("visionbrain.live_tracking")

DEFAULT_MODEL = "mlx-community/sam3.1-bf16"
DEFAULT_RESOLUTION = 1008
DEFAULT_DETECT_EVERY = 6
DEFAULT_BACKBONE_EVERY = 15
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MAX_LOST = 10
DEFAULT_MAX_BACKBONE_AGE_MS = 3000
_U32 = 1 << 32

_lock = threading.Lock()
_loaded: dict[tuple[str, int], dict[str, Any]] = {}


def _ensure_loaded(model: str, resolution: int, threshold: float):
    """Lazy-load SAM 3.1 model/processor/predictor once per (model, res).

    ``threshold`` is re-applied to the predictor on a cache hit too — a later
    tracker built with a different threshold must not silently run at the
    first-load value.
    """
    global _loaded
    key = (model, resolution)
    with _lock:
        if key not in _loaded:
            from mlx_vlm.models.sam3_1.generate import Sam3Predictor
            from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
            from mlx_vlm.utils import get_model_path, load_model

            log.info("loading SAM 3.1 %s (res=%s) for tracking...", model, resolution)
            mp = get_model_path(model)
            m = load_model(mp)
            proc = Sam31Processor.from_pretrained(str(mp))
            if resolution != 1008:
                proc.image_size = resolution
            pred = Sam3Predictor(m, proc, score_threshold=threshold)
            _loaded[key] = {"model": m, "processor": proc, "predictor": pred}
            log.info("SAM 3.1 tracking loaded")
        entry = _loaded[key]
        predictor = entry["predictor"]
        try:
            predictor.score_threshold = float(threshold)
        except AttributeError:
            pass  # injected test double without predictor attributes
        return entry["model"], entry["processor"], predictor


def _elapsed_ms(now_ms: int, then_ms: int) -> int:
    """Source-clock difference; timestamps are wrapped u32 milliseconds."""
    return (int(now_ms) - int(then_ms)) % _U32


def _default_backbone(model, pixel_values):
    from mlx_vlm.models.sam3_1.generate import _get_backbone_features

    return _get_backbone_features(model, pixel_values)


def _default_detect(predictor, backbone_features, prompts, image_size, threshold, encoder_cache):
    from mlx_vlm.models.sam3_1.generate import _detect_with_backbone

    return _detect_with_backbone(
        predictor,
        backbone_features,
        prompts,
        image_size,
        threshold,
        encoder_cache=encoder_cache,
    )


def _default_preprocess(processor, image):
    """Processor-specific resize/normalize -> MLX array for the backbone."""
    import mlx.core as mx

    inputs = processor.preprocess_image(image)
    return mx.array(inputs["pixel_values"])


def _default_mask_to_polygon(mask, width: int, height: int):
    """Default mask->polygon hook (detection_core's dependency-free tracer)."""
    from .detection_core import mask_to_polygon

    return mask_to_polygon(mask, width, height)


class LiveSamTracker:
    """Per-shot SAM 3.1 tracker holding backbone cache + SimpleTracker state.

    Call :meth:`step` once per frame; returns normalized items carrying
    ``label`` / ``score`` / ``box`` (xyxy normalized 0-1) / ``source="sam"`` /
    ``track_id`` / ``color_id`` / ``track_state`` (+ ``polygon`` — the mask
    outline, extracted by default whenever the model returns masks).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        resolution: int = DEFAULT_RESOLUTION,
        threshold: float = 0.15,
        detect_every: int = DEFAULT_DETECT_EVERY,
        backbone_every: int = DEFAULT_BACKBONE_EVERY,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_lost: int = DEFAULT_MAX_LOST,
        max_backbone_age_ms: int = DEFAULT_MAX_BACKBONE_AGE_MS,
        backbone_fn: Optional[Callable] = None,
        detect_fn: Optional[Callable] = None,
        tracker=None,
        mask_to_polygon=None,
        preprocess_fn: Optional[Callable] = None,
    ) -> None:
        self.model = model
        self.resolution = int(resolution)
        self.threshold = float(threshold)
        self.detect_every = max(1, int(detect_every))
        self.backbone_every = max(1, int(backbone_every))
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost
        self.max_backbone_age_ms = int(max_backbone_age_ms)
        self._backbone_fn = backbone_fn or _default_backbone
        self._detect_fn = detect_fn or _default_detect
        self._preprocess_fn = preprocess_fn or _default_preprocess
        # Polygon outlines are emitted by default so every consumer sees the
        # painted mask shape, not just the box around it; callers may still
        # inject their own tracer (or None to disable).
        self._mask_to_polygon = mask_to_polygon or _default_mask_to_polygon
        self._injected_tracker = tracker  # injectable; else lazily built
        self.reset()

    def reset(self) -> None:
        """Clear tracker state (new shot/scene); backbone cache is dropped."""
        self._tracker = self._injected_tracker
        if self._tracker is None:
            try:
                from mlx_vlm.models.sam3.generate import SimpleTracker

                self._tracker = SimpleTracker(
                    iou_threshold=self._iou_threshold, max_lost=self._max_lost
                )
            except Exception:
                self._tracker = None
        self._detect_ticks = 0
        self._detect_passes = 0
        self._backbone_cache = None
        self._backbone_frame_id: int | None = None
        self._backbone_timestamp_ms = 0
        self._config_key: Any = None
        self._encoder_cache: dict[str, Any] = {}
        self._last_items: list[dict[str, Any]] = []

    def step(
        self,
        image,
        prompts: list[str],
        task: str,
        width: int,
        height: int,
        frame_id: int,
        timestamp_ms: int,
        source_key: Any = None,
    ) -> list[dict[str, Any]]:
        """Run one tracking step; returns normalized detection items.

        ``source_key`` identifies the producer (id and epoch). Any change to it,
        the prompts, task, threshold or frame size drops the held items and the
        cached backbone.
        """
        config_key = (
            tuple(prompts), task, self.threshold, int(width), int(height), source_key
        )
        if config_key != self._config_key:
            if self._config_key is not None:
                self.reset()
            self._config_key = config_key
        do_detect = self._detect_ticks % self.detect_every == 0

        if not do_detect:
            # Held re-publish — even when the last detect found nothing
            # (empty set held as empty): an empty scene must not defeat the
            # detect_every throttle by forcing a full detect every frame.
            # stale_ms uses the source's own clock so phone-vs-Mac skew cancels;
            # reporting 0 would be a lie with teeth — downstream evidence
            # filtering treats stale_ms == 0 as "current measurement".
            held = []
            for it in self._last_items:
                d = dict(it)
                observed = d.get("observed_timestamp_ms")
                observed = timestamp_ms if observed is None else observed
                d["stale_ms"] = _elapsed_ms(timestamp_ms, observed)
                d["track_state"] = "predicted"
                held.append(d)
            self._detect_ticks += 1
            return held

        model, processor, predictor = _ensure_loaded(
            self.model, self.resolution, self.threshold
        )

        # Recompute the ViT backbone only every backbone_every detect passes,
        # or once the cached one is too old.
        if (
            self._backbone_cache is None
            or self._detect_passes % self.backbone_every == 0
            or _elapsed_ms(timestamp_ms, self._backbone_timestamp_ms)
            > self.max_backbone_age_ms
        ):
            pixels = self._preprocess_fn(processor, image)
            self._backbone_cache = self._backbone_fn(model, pixels)
            self._backbone_frame_id = frame_id
            self._backbone_timestamp_ms = timestamp_ms
        backbone_age_ms = _elapsed_ms(timestamp_ms, self._backbone_timestamp_ms)

        self._encoder_cache.clear()
        result = self._detect_fn(
            predictor,
            self._backbone_cache,
            prompts,
            (width, height),
            self.threshold,
            self._encoder_cache,
        )
        result = self._tracker.update(result) if self._tracker is not None else result

        items: list[dict[str, Any]] = []
        scores = getattr(result, "scores", None)
        boxes = getattr(result, "boxes", None)
        labels = getattr(result, "labels", None) or []
        masks = getattr(result, "masks", None)
        track_ids = getattr(result, "track_ids", None)
        n = len(scores) if scores is not None else 0

        for i in range(n):
            box = list(boxes[i])
            if len(box) < 4:
                continue
            norm = [
                round(max(0.0, min(float(box[0]) / width, 1.0)), 4),
                round(max(0.0, min(float(box[1]) / height, 1.0)), 4),
                round(max(0.0, min(float(box[2]) / width, 1.0)), 4),
                round(max(0.0, min(float(box[3]) / height, 1.0)), 4),
            ]
            if norm[2] <= norm[0] or norm[3] <= norm[1]:
                continue
            tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else i
            label = str(labels[i]) if i < len(labels) else "object"
            item: dict[str, Any] = {
                "label": label,
                "score": round(float(scores[i]), 4),
                "box": norm,
                "source": "sam",
                "track_id": tid,
                "color_id": tid % 8,
                "track_state": "active" if backbone_age_ms == 0 else "predicted",
                "track_confidence": round(float(scores[i]), 4),
                "observed_frame_id": self._backbone_frame_id,
                "observed_timestamp_ms": self._backbone_timestamp_ms,
                "stale_ms": backbone_age_ms,
            }
            if task == "segment" and masks is not None and i < len(masks):
                mask = masks[i]
                if self._mask_to_polygon is not None:
                    polygon = self._mask_to_polygon(mask, width, height)
                    if polygon:
                        item["polygon"] = polygon
            items.append(item)

        self._last_items = items
        self._detect_ticks += 1
        self._detect_passes += 1
        return items
