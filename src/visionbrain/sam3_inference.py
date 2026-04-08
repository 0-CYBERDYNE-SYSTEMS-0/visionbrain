"""SAM 3.1 inference — video object tracking and multi-prompt segmentation.

Requires facebook/sam3.1 weights to be cached locally (~3GB).
Run once:  huggingface-cli download facebook/sam3.1

This module wraps the SAM 3.1 implementation inside mlx_vlm, which in turn
calls the Meta Segment Anything Model 3.1 with TriViTDetNeck and
Object Multiplex tracking.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from .loader import sam31_cache_path, _check_mlx

# ──────────────────────────────────────────────────────────────────────────────
# Availability check
# ──────────────────────────────────────────────────────────────────────────────

def sam31_available() -> bool:
    """True if SAM 3.1 can run on this machine."""
    if not _check_mlx():
        return False
    return sam31_cache_path() is not None


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (cached per process)
# ──────────────────────────────────────────────────────────────────────────────

_sam_model_cache: dict = {}


def _ensure_sam31(
    model_path: Optional[str] = None,
    threshold: float = 0.15,
    resolution: int = 1008,
):
    """Load SAM 3.1 model + processor once."""
    if "model" not in _sam_model_cache:
        if not sam31_available():
            raise RuntimeError(
                "SAM 3.1 weights not cached. Run:\n"
                "  huggingface-cli download facebook/sam3.1\n"
                "Then restart this session."
            )

        from mlx_vlm.utils import get_model_path, load_model
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.models.sam3_1.generate import Sam3Predictor

        hf_repo = model_path or "facebook/sam3.1"
        print(f"Loading SAM 3.1 from {hf_repo}...")
        t0 = time.perf_counter()
        mp = get_model_path(hf_repo)
        model = load_model(mp)
        processor = Sam31Processor.from_pretrained(str(mp))
        if resolution != 1008:
            processor.image_size = resolution
        predictor = Sam3Predictor(model, processor, score_threshold=threshold)
        print(f"  Loaded in {time.perf_counter()-t0:.2f}s")

        _sam_model_cache["model"] = model
        _sam_model_cache["processor"] = processor
        _sam_model_cache["predictor"] = predictor

    return (
        _sam_model_cache["model"],
        _sam_model_cache["processor"],
        _sam_model_cache["predictor"],
    )


# ──────────────────────────────────────────────────────────────────────────────
# Single-image multi-prompt detection + segmentation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Sam31Detection:
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]   # x1, y1, x2, y2 (pixels)
    mask: Optional[np.ndarray] = None                # (H, W) uint8, if segmentation

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(float(self.score), 4),
            "bbox_xyxy": [round(float(x), 4) for x in self.bbox_xyxy],
            "has_mask": self.mask is not None,
        }


def detect_multi(
    image: Image.Image,
    prompts: list[str],
    *,
    threshold: float = 0.15,
    resolution: int = 1008,
    task: str = "detect",
) -> list[Sam31Detection]:
    """Run SAM 3.1 multi-prompt detection/segmentation on a still image.

    Runs the vision backbone ONCE, then DETR per prompt.
    Cost = 1x ViT + Nx (text + DETR) instead of Nx full pipeline.

    Args:
        image: PIL Image
        prompts: list of text prompts, e.g. ["cow", "sheep", "fence"]
        threshold: confidence threshold (lower = more detections)
        resolution: input resolution (1008 = native)
        task: "detect" (bboxes only) or "segment" (with masks)

    Returns:
        list of Sam31Detection, one per found object
    """
    from mlx_vlm.models.sam3_1.generate import predict_multi

    model, processor, predictor = _ensure_sam31(threshold=threshold, resolution=resolution)

    t0 = time.perf_counter()
    result = predict_multi(
        predictor=predictor,
        image=image,
        prompts=prompts,
        score_threshold=threshold,
    )
    print(f"  SAM 3.1 detect_multi: {len(result.scores)} detections in {time.perf_counter()-t0:.2f}s")

    detections = []
    img_w, img_h = image.size
    for i, (score, box_xyxy, label) in enumerate(zip(result.scores, result.boxes, result.labels or prompts * len(result.scores))):
        mask = None
        if task == "segment" and result.masks is not None and i < len(result.masks):
            mask = result.masks[i]

        detections.append(Sam31Detection(
            label=label or "object",
            score=float(score),
            bbox_xyxy=(float(box_xyxy[0]), float(box_xyxy[1]),
                        float(box_xyxy[2]), float(box_xyxy[3])),
            mask=mask,
        ))
    return detections


# ──────────────────────────────────────────────────────────────────────────────
# Video tracking
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoTrackStats:
    total_frames: int
    processed_frames: int
    fps: float
    unique_objects: int
    output_path: str


def track_video(
    video_path: str,
    prompts: list[str],
    output_path: Optional[str] = None,
    *,
    threshold: float = 0.15,
    every_n_frames: int = 2,
    backbone_every: int = 1,
    show_boxes: bool = True,
    resolution: int = 1008,
    opacity: float = 0.6,
    contour_thickness: int = 2,
) -> VideoTrackStats:
    """Track objects in a video file using SAM 3.1.

    Args:
        video_path: path to video file
        prompts: text prompts to track, e.g. ["cow", "horse"]
        output_path: output video path (auto-generated if None)
        threshold: detection confidence
        every_n_frames: run DETR detection every N frames (1 = every frame)
        backbone_every: re-run ViT backbone every N detections
        show_boxes: draw bounding boxes on output
        resolution: input resolution (1008 = native, lower = faster)
        opacity: mask overlay opacity (0-1)
        contour_thickness: contour line thickness

    Returns:
        VideoTrackStats with timing and counts
    """
    from mlx_vlm.models.sam3_1.generate import track_video as _track_video

    if output_path is None:
        p = Path(video_path)
        output_path = str(p.parent / f"{p.stem}_tracked{p.suffix}")

    t0 = time.perf_counter()
    _track_video(
        video_path=video_path,
        prompts=prompts,
        output=output_path,
        threshold=threshold,
        every=every_n_frames,
        show_boxes=show_boxes,
        resolution=resolution,
        backbone_every=backbone_every,
        opacity=opacity,
        contour_thickness=contour_thickness,
    )

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    elapsed = time.perf_counter() - t0
    processed = total // every_n_frames

    print(f"  Video tracking complete: {processed}/{total} frames in {elapsed:.1f}s ({fps:.1f} fps display)")

    return VideoTrackStats(
        total_frames=total,
        processed_frames=processed,
        fps=round(fps, 1),
        unique_objects=len(prompts),
        output_path=output_path,
    )


def track_realtime(
    camera_or_video: str,
    prompts: list[str],
    *,
    threshold: float = 0.15,
    detect_every: int = 15,
    recompute_backbone_every: int = 30,
    update_memory_every: int = 3,
    resolution: int = 1008,
) -> None:
    """Real-time tracking from camera (0) or video file.

    Optimizations:
    - Backbone caching: skip ViT on intermediate frames (~67ms saved per frame)
    - Tracker propagation: use memory attention + mask decoder instead of DETR
    - Only re-runs DETR every detect_every frames

    Press 'q' in the display window to quit.

    Args:
        camera_or_video: "0" for webcam, or path to video file
        prompts: text prompts to track
        threshold: detection confidence
        detect_every: run DETR detection every N inference frames
        recompute_backbone_every: re-run ViT backbone every N frames
        update_memory_every: update tracker memory every N propagation frames
        resolution: input resolution (1008 = native)
    """
    from mlx_vlm.models.sam3_1.generate import track_video_realtime as _track_realtime

    if not sam31_available():
        raise RuntimeError(
            "SAM 3.1 weights not cached. Run:\n"
            "  huggingface-cli download facebook/sam3.1\n"
            "Then restart this session."
        )

    _track_realtime(
        video_path=camera_or_video,
        prompts=prompts,
        threshold=threshold,
        detect_every=detect_every,
        recompute_backbone_every=recompute_backbone_every,
        update_memory_every=update_memory_every,
        resolution=resolution,
    )
