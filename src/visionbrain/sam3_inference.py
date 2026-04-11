"""SAM 3.1 inference — video object tracking and multi-prompt segmentation.

Weights: mlx-community/sam3.1-bf16 (~3GB, public download).
Loads via mlx_vlm's standard MLX loader — no gated access needed.
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

from .loader import sam31_cache_path, _check_mlx, SAM31_HF_REPO

HF_REPO = SAM31_HF_REPO  # "mlx-community/sam3.1-bf16"

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
                "  huggingface-cli download mlx-community/sam3.1-bf16\n"
                "Then restart this session."
            )

        from mlx_vlm.utils import get_model_path, load_model
        from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
        from mlx_vlm.models.sam3_1.generate import Sam3Predictor

        t0 = time.perf_counter()
        hf_repo = model_path or HF_REPO
        print(f"Loading SAM 3.1 from {hf_repo}...")
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
    model_path: str = "mlx-community/sam3.1-bf16",
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
        model_path: HuggingFace repo ID for SAM 3.1 weights
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
        model_path=model_path,
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


# ──────────────────────────────────────────────────────────────────────────────
# track_video_with_json — tracking + detection export + annotated video
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FrameDetection:
    """One object detection in one frame."""
    label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    track_id: int
    centroid_norm: tuple[float, float]  # (x_norm, y_norm) relative to image
    area_fraction: float                 # fraction of image area

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(float(self.score), 4),
            "track_id": int(self.track_id),
            "bbox_xyxy": [round(float(x), 2) for x in self.bbox_xyxy],
            "centroid_norm": {"x": round(float(self.centroid_norm[0]), 4),
                              "y": round(float(self.centroid_norm[1]), 4)},
            "area_fraction": round(float(self.area_fraction), 5),
        }


def track_video_with_json(
    video_path: str,
    prompts: list[str],
    output_path: Optional[str] = None,
    json_path: Optional[str] = None,
    *,
    model_path: str = "mlx-community/sam3.1-bf16",
    threshold: float = 0.15,
    every_n_frames: int = 2,
    backbone_every: int = 1,
    resolution: int = 1008,
    opacity: float = 0.6,
    contour_thickness: int = 2,
) -> tuple[VideoTrackStats, list[dict]]:
    """Track objects in a video + export per-frame detections as JSON.

    This is the pipeline-grade version of track_video(). It captures every
    detection with bounding boxes, track IDs, normalized centroids, and area
    fractions — all the structured data Gemma 4 needs to reason intelligently.

    Args:
        video_path: input video path
        prompts: text prompts to track, e.g. ["cow", "sheep", "fence"]
        output_path: annotated video output (auto-generated if None)
        json_path: JSON detections output (auto-generated if None)
        model_path: HuggingFace repo ID for SAM 3.1 weights
        threshold: detection confidence
        every_n_frames: run DETR detection every N frames
        backbone_every: re-run ViT backbone every N detections
        resolution: SAM input resolution (1008 = native)
        opacity: mask overlay opacity
        contour_thickness: contour line thickness

    Returns:
        (VideoTrackStats, list of per-frame detection dicts ready for Gemma 4)
    """
    import json as _json
    import mlx.core as mx
    from PIL import Image

    from mlx_vlm.models.sam3_1.processing_sam3_1 import Sam31Processor
    from mlx_vlm.models.sam3_1.generate import (
        Sam3Predictor,
        SimpleTracker,
        _get_backbone_features,
        _detect_with_backbone,
        draw_frame,
    )

    p = Path(video_path)
    if output_path is None:
        output_path = str(p.parent / f"{p.stem}_tracked{p.suffix}")
    if json_path is None:
        json_path = str(p.parent / f"{p.stem}_detections.json")

    print(f"Loading SAM 3.1 from {model_path}...")
    from mlx_vlm.utils import get_model_path, load_model
    mp = get_model_path(model_path)
    model = load_model(mp)
    processor = Sam31Processor.from_pretrained(str(mp))
    if resolution != 1008:
        processor.image_size = resolution
    predictor = Sam3Predictor(model, processor, score_threshold=threshold)
    tracker = SimpleTracker(iou_threshold=0.3, max_lost=10)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {total_frames} frames, {fps:.1f} fps, {W}x{H}")
    print(f"Tracking: {prompts}, every {every_n_frames} frames, threshold {threshold}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    backbone_cache = None
    encoder_cache = {}
    latest_result = None

    all_frames: list[dict] = []
    detect_count = 0
    t_start = time.perf_counter()

    for fi in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        is_detect_frame = (fi % every_n_frames == 0)

        if is_detect_frame:
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            inputs = processor.preprocess_image(frame_pil)
            pixel_values = mx.array(inputs["pixel_values"])

            if detect_count % backbone_every == 0 or backbone_cache is None:
                backbone_cache = _get_backbone_features(model, pixel_values)
                encoder_cache.clear()

            result = _detect_with_backbone(
                predictor,
                backbone_cache,
                prompts,
                frame_pil.size,
                threshold,
                encoder_cache=encoder_cache,
            )
            latest_result = tracker.update(result)
            detect_count += 1

        # Annotate frame (use latest result or empty result for non-detect frames)
        if latest_result is not None and len(latest_result.scores) > 0:
            out = draw_frame(
                frame_bgr,
                latest_result.masks,
                latest_result.scores,
                latest_result.boxes,
                " + ".join(prompts),
                H, W,
                show_boxes=True,
                labels=latest_result.labels,
            )
        else:
            out = frame_bgr

        writer.write(out)

        # Collect frame detection data
        if is_detect_frame and latest_result is not None:
            frame_data = {
                "frame_index": fi,
                "timestamp": round(fi / fps, 3),
                "n_detections": len(latest_result.scores),
                "detections": [],
            }
            img_area = W * H

            scores = latest_result.scores
            boxes = latest_result.boxes
            labels = latest_result.labels or (prompts * len(scores))
            track_ids = getattr(latest_result, "track_ids", None)

            for i, (score, box, label) in enumerate(zip(scores, boxes, labels)):
                bx1, by1, bx2, by2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                cx_norm = ((bx1 + bx2) / 2) / W
                cy_norm = ((by1 + by2) / 2) / H
                box_area = (bx2 - bx1) * (by2 - by1)
                area_frac = box_area / img_area
                tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else i

                det = FrameDetection(
                    label=label or "object",
                    score=float(score),
                    bbox_xyxy=(bx1, by1, bx2, by2),
                    track_id=tid,
                    centroid_norm=(cx_norm, cy_norm),
                    area_fraction=area_frac,
                )
                frame_data["detections"].append(det.to_dict())

            all_frames.append(frame_data)

        if fi % 40 == 0 and fi > 0:
            elapsed = time.perf_counter() - t_start
            fps_actual = (fi + 1) / elapsed if elapsed > 0 else 0
            n_dets = len(latest_result.scores) if latest_result else 0
            print(f"  Frame {fi}/{total_frames}: {n_dets} det, {fps_actual:.1f} fps")

    writer.release()
    cap.release()
    elapsed = time.perf_counter() - t_start

    # Write JSON
    with open(json_path, "w") as f:
        _json.dump({
            "video_path": str(video_path),
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "resolution": f"{W}x{H}",
            "prompts": prompts,
            "threshold": threshold,
            "processed_frames": len(all_frames),
            "elapsed_seconds": round(elapsed, 1),
            "frames": all_frames,
        }, f, indent=2)

    # Compute unique track IDs
    all_track_ids = set()
    for frame in all_frames:
        for det in frame["detections"]:
            all_track_ids.add(det["track_id"])

    print(f"\nSaved: {output_path}")
    print(f"Saved: {json_path}  ({len(all_frames)} frames, {len(all_track_ids)} unique objects)")

    stats = VideoTrackStats(
        total_frames=total_frames,
        processed_frames=len(all_frames),
        fps=round(fps, 1),
        unique_objects=len(all_track_ids),
        output_path=output_path,
    )
    return stats, all_frames


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
            "  huggingface-cli download mlx-community/sam3.1-bf16\n"
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
