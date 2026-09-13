"""Supervision bridge — convert VisionBrain model outputs to sv.Detections.

Provides interoperability between VisionBrain's custom dataclasses
(Sam31Detection, MaskResult, DetectionResult) and the Supervision
ecosystem (ByteTrack, annotators, zone analytics, CompactMask).
"""

from __future__ import annotations

import warnings
import weakref
from typing import Optional

import numpy as np
from PIL import Image

# Supervision imports
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack as _RealByteTrack

from .fp_inference import MaskResult, DetectionResult
from .sam3_inference import Sam31Detection


# ──────────────────────────────────────────────────────────────────────────────
# ByteTrack wrapper (hides deprecation warning, future-proofs against 0.30)
# ──────────────────────────────────────────────────────────────────────────────

class ByteTrack:
    """VisionBrain wrapper around Supervision's ByteTrack.

    Hides the FutureWarning deprecation noise and provides a stable
    interface regardless of whether ByteTrack moves to a new module
    in supervision 0.30+.
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = _RealByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate,
                minimum_consecutive_frames=minimum_consecutive_frames,
            )

    def update_with_detections(self, detections: sv.Detections) -> sv.Detections:
        return self._tracker.update_with_detections(detections)

    def reset(self) -> None:
        self._tracker.reset()


# ──────────────────────────────────────────────────────────────────────────────
# Conversion: Sam31Detection → sv.Detections
# ──────────────────────────────────────────────────────────────────────────────

def detections_from_sam31(
    detections: list[Sam31Detection],
    image_shape: tuple[int, int],
    class_id_offset: int = 0,
) -> sv.Detections:
    """Convert SAM 3.1 detections to Supervision format.

    Args:
        detections: List of Sam31Detection from detect_multi()
        image_shape: (height, width) of the source image
        class_id_offset: Added to each class_id for prompt indexing

    Returns:
        sv.Detections with xyxy, mask, confidence, class_id
    """
    if not detections:
        return sv.Detections.empty()

    h, w = image_shape
    n = len(detections)

    xyxy = np.zeros((n, 4), dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    class_id = np.zeros(n, dtype=int)

    masks = []
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det.bbox_xyxy
        xyxy[i] = [x1, y1, x2, y2]
        confidence[i] = det.score
        class_id[i] = class_id_offset + i  # prompt index as class_id

        if det.mask is not None:
            # Ensure mask is (H, W) bool
            mask = det.mask
            if mask.shape != (h, w):
                mask = np.array(
                    Image.fromarray(mask.astype(np.uint8), mode="L").resize((w, h), Image.NEAREST)
                )
            masks.append(mask.astype(bool))
        else:
            masks.append(np.zeros((h, w), dtype=bool))

    mask_array = np.stack(masks) if masks else None

    return sv.Detections(
        xyxy=xyxy,
        mask=mask_array,
        confidence=confidence,
        class_id=class_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Conversion: Falcon Perception masks → sv.Detections
# ──────────────────────────────────────────────────────────────────────────────

def detections_from_falcon_masks(
    masks: list[MaskResult],
    image_shape: tuple[int, int],
) -> sv.Detections:
    """Convert Falcon Perception segmentation masks to Supervision format.

    Args:
        masks: List of MaskResult from segment()
        image_shape: (height, width) of the source image

    Returns:
        sv.Detections with xyxy, mask, confidence=1.0, class_id=0
    """
    if not masks:
        return sv.Detections.empty()

    h, w = image_shape
    n = len(masks)

    xyxy = np.zeros((n, 4), dtype=np.float32)
    confidence = np.ones(n, dtype=np.float32)
    class_id = np.zeros(n, dtype=int)

    mask_arrays = []
    for i, m in enumerate(masks):
        # Convert normalized bbox to pixel xyxy
        x1 = m.bbox_x1 * w
        y1 = m.bbox_y1 * h
        x2 = m.bbox_x2 * w
        y2 = m.bbox_y2 * h
        xyxy[i] = [x1, y1, x2, y2]

        # Decode RLE to boolean mask
        from pycocotools import mask as mask_utils

        rle = m.rle.copy()
        if isinstance(rle.get("counts"), str):
            rle["counts"] = rle["counts"].encode("utf-8")
        binary = mask_utils.decode(rle).astype(bool)

        if binary.shape != (h, w):
            binary = np.array(
                Image.fromarray(binary.astype(np.uint8), mode="L").resize((w, h), Image.NEAREST)
            ).astype(bool)

        mask_arrays.append(binary)

    mask_array = np.stack(mask_arrays)

    return sv.Detections(
        xyxy=xyxy,
        mask=mask_array,
        confidence=confidence,
        class_id=class_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Conversion: Falcon Perception boxes → sv.Detections
# ──────────────────────────────────────────────────────────────────────────────

def detections_from_falcon_boxes(
    detections: list[DetectionResult],
    image_shape: tuple[int, int],
) -> sv.Detections:
    """Convert Falcon Perception bounding-box detections to Supervision format.

    Args:
        detections: List of DetectionResult from detect()
        image_shape: (height, width) of the source image

    Returns:
        sv.Detections with xyxy, confidence, class_id (no masks)
    """
    if not detections:
        return sv.Detections.empty()

    h, w = image_shape
    n = len(detections)

    xyxy = np.zeros((n, 4), dtype=np.float32)
    confidence = np.zeros(n, dtype=np.float32)
    class_id = np.zeros(n, dtype=int)

    for i, det in enumerate(detections):
        cx, cy, bh, bw = det.cx, det.cy, det.h, det.w
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        xyxy[i] = [x1, y1, x2, y2]
        confidence[i] = det.score

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CompactMask conversion
# ──────────────────────────────────────────────────────────────────────────────

def to_compact(detections: sv.Detections, image_shape: tuple[int, int] | None = None) -> sv.Detections:
    """Replace full-resolution masks with sv.CompactMask for memory efficiency.

    Returns a new Detections object; original is unchanged.
    """
    if detections.mask is None:
        return detections

    # CompactMask stores crop + RLE per mask
    # Use from_dense (the correct API in supervision 0.28)
    h, w = image_shape or detections.data.get("image_shape", (0, 0))
    if h == 0 or w == 0:
        # Infer from mask shape
        h, w = detections.mask.shape[1], detections.mask.shape[2]
    compact = sv.CompactMask.from_dense(
        detections.mask, detections.xyxy, (h, w)
    )

    # Build new Detections with compact mask reference
    # CompactMask can't go in `data` dict (supervision validates it)
    # Store it in a module-level cache keyed by the returned Detections object's id
    new_data = dict(detections.data)

    # Create the new detections object first
    result = sv.Detections(
        xyxy=detections.xyxy.copy(),
        confidence=detections.confidence.copy() if detections.confidence is not None else None,
        class_id=detections.class_id.copy() if detections.class_id is not None else None,
        tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None,
        data=new_data,
    )

    # Store compact mask keyed by the result object's id (so from_compact can
    # find it); a finalizer evicts the entry when the result is collected —
    # without it the cache grows forever and a recycled id() would hand an
    # unrelated object's masks to from_compact.
    _compact_mask_cache[id(result)] = compact
    _compact_shape_cache[id(result)] = (h, w)
    weakref.finalize(result, _evict_compact_cache, id(result))

    return result


def _evict_compact_cache(key: int) -> None:
    """Drop both cache entries for one collected Detections object."""
    _compact_mask_cache.pop(key, None)
    _compact_shape_cache.pop(key, None)


# Module-level caches for compact masks and shapes (since they can't live in Detections.data)
_compact_mask_cache: dict[int, sv.CompactMask] = {}
_compact_shape_cache: dict[int, tuple[int, int]] = {}


def from_compact(detections: sv.Detections) -> sv.Detections:
    """Materialize CompactMask back to full np.ndarray masks."""
    # Find any compact mask in our cache that matches these detections
    # We use the detection object's id as a lookup key
    det_id = id(detections)
    compact = _compact_mask_cache.get(det_id)
    if compact is None:
        return detections
    # to_dense returns (N, H, W) boolean array
    full_masks = compact.to_dense()
    return sv.Detections(
        xyxy=detections.xyxy.copy(),
        confidence=detections.confidence.copy() if detections.confidence is not None else None,
        class_id=detections.class_id.copy() if detections.class_id is not None else None,
        tracker_id=detections.tracker_id.copy() if detections.tracker_id is not None else None,
        mask=full_masks,
        data=dict(detections.data),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Backward compatibility: sv.Detections → legacy dict
# ──────────────────────────────────────────────────────────────────────────────

def to_legacy_dict(
    detections: sv.Detections,
    labels: Optional[list[str]] = None,
) -> list[dict]:
    """Convert sv.Detections to VisionBrain's legacy JSON-serializable format.

    Args:
        detections: Supervision detections
        labels: Optional label names per class_id (defaults to class_id as string)

    Returns:
        List of detection dicts matching the old Sam31Detection.to_dict() shape
    """
    results = []
    n = len(detections)

    for i in range(n):
        x1, y1, x2, y2 = detections.xyxy[i]
        det = {
            "label": labels[detections.class_id[i]] if labels else str(detections.class_id[i]),
            "score": round(float(detections.confidence[i]), 4) if detections.confidence is not None else 1.0,
            "bbox_xyxy": [round(float(x1), 4), round(float(y1), 4),
                          round(float(x2), 4), round(float(y2), 4)],
            "has_mask": detections.mask is not None,
        }
        if detections.tracker_id is not None:
            det["tracker_id"] = int(detections.tracker_id[i])
        results.append(det)

    return results
