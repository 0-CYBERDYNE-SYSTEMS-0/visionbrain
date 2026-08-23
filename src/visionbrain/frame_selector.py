"""Fast frame scorer — Falcon-based relevance filter for large videos.

Extracts a small number of frames, runs Falcon Perception at low resolution
to score relevance to the query, and returns a quick answer + temporal
regions of interest before the full SAM pipeline runs.

Used by the --fast flag on analyze and by the fastscan CLI command.
"""

from __future__ import annotations

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FrameScore:
    frame_index: int
    timestamp: float
    relevance_score: float  # 0-1
    detection_count: int
    top_label: str
    has_query_match: bool


@dataclass
class TemporalRegion:
    start_time: float
    end_time: float
    avg_relevance: float
    label: str


@dataclass
class FrameScores:
    """Output of score_frames()."""

    video_path: str
    total_frames: int
    fps: float
    duration_s: float
    frames_scored: int
    is_relevant: bool
    quick_answer: str
    regions: list[TemporalRegion] = field(default_factory=list)
    frame_scores: list[FrameScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration_s": round(self.duration_s, 2),
            "frames_scored": self.frames_scored,
            "is_relevant": self.is_relevant,
            "quick_answer": self.quick_answer,
            "regions": [
                {
                    "start": round(r.start_time, 2),
                    "end": round(r.end_time, 2),
                    "avg_relevance": round(r.avg_relevance, 3),
                    "label": r.label,
                }
                for r in self.regions
            ],
            "frame_scores": [
                {
                    "frame": f.frame_index,
                    "t": round(f.timestamp, 2),
                    "score": round(f.relevance_score, 3),
                    "dets": f.detection_count,
                    "label": f.top_label,
                }
                for f in self.frame_scores
            ],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Core scoring function
# ──────────────────────────────────────────────────────────────────────────────


def score_frames(
    video_path: str,
    query: str,
    *,
    sample_every_n_seconds: float = 5.0,
    max_frames: int = 60,
    resolution: int = 360,
    min_relevance: float = 0.2,
) -> FrameScores:
    """Score video frames by relevance to a query using Falcon Perception.

    Extracts frames at uniform intervals (default: every 5s, max 60 frames),
    runs Falcon detect at low resolution (default 360p), scores each frame
    by detection count and query match, clusters high-scoring frames into
    temporal regions, and returns a quick natural-language answer.

    Args:
        video_path: Path to the video file.
        query: Natural-language query, e.g. "vehicles near the gate".
        sample_every_n_seconds: Sample one frame every N seconds (default 5).
        max_frames: Maximum number of frames to score (default 60).
        resolution: Resolution to run Falcon at (default 360 — lower = faster).
        min_relevance: Minimum relevance score to count as a region (default 0.2).

    Returns:
        FrameScores with quick answer, temporal regions, and per-frame scores.
    """
    from .loader import falcon_perception_record
    from .fp_inference import detect

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_s = total_frames / fps if fps > 0 else 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rec = falcon_perception_record()
    if not rec.can_load:
        raise RuntimeError(
            f"Falcon Perception not ready ({rec.note}). "
            "Cannot run fast-path scan."
        )

    # Determine which frame indices to sample
    sample_interval = max(1, int(sample_every_n_seconds * fps))
    candidate_indices = list(range(0, total_frames, sample_interval))
    if len(candidate_indices) > max_frames:
        candidate_indices = candidate_indices[:max_frames]

    t_start = time.perf_counter()
    scored_frames: list[FrameScore] = []
    query_terms = set(query.lower().split())

    for frame_idx in candidate_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue

        # Convert to PIL at low resolution for Falcon
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_small = Image.fromarray(frame_rgb)

        # Scale down to target resolution (preserving aspect ratio)
        scale = resolution / max(W, H)
        new_w = int(W * scale)
        new_h = int(H * scale)
        pil_small = pil_small.resize((new_w, new_h), Image.LANCZOS)

        timestamp = frame_idx / fps if fps > 0 else 0

        # Run Falcon detection
        try:
            results, stats = detect(
                pil_small,
                query,
                max_new_tokens=100,  # Low token budget for speed
            )
        except Exception:
            # If Falcon fails on a frame, skip it gracefully
            scored_frames.append(
                FrameScore(
                    frame_index=frame_idx,
                    timestamp=timestamp,
                    relevance_score=0.0,
                    detection_count=0,
                    top_label="",
                    has_query_match=False,
                )
            )
            continue

        # Score the frame
        detection_count = len(results)
        top_label = results[0].label if results else ""

        # Compute relevance: weighted combination of detection count and label match
        # Higher detection count = more relevant
        norm_dets = min(detection_count / 5.0, 1.0)  # normalize to 5 det = max

        # Check if Falcon's returned labels match query terms
        label_match = 0.0
        if top_label:
            label_terms = set(top_label.lower().split())
            overlap = query_terms & label_terms
            label_match = len(overlap) / max(len(query_terms), 1)

        relevance = norm_dets * 0.7 + label_match * 0.3
        relevance = min(relevance, 1.0)

        scored_frames.append(
            FrameScore(
                frame_index=frame_idx,
                timestamp=timestamp,
                relevance_score=relevance,
                detection_count=detection_count,
                top_label=top_label,
                has_query_match=relevance >= min_relevance,
            )
        )

    cap.release()

    elapsed = time.perf_counter() - t_start

    # Determine if video is relevant at all
    is_relevant = any(f.relevance_score >= min_relevance for f in scored_frames)

    # Cluster high-scoring frames into temporal regions
    regions = _cluster_regions(scored_frames, min_relevance=min_relevance)

    # Build quick answer
    quick_answer = _build_quick_answer(
        scored_frames, regions, query, is_relevant, duration_s, fps
    )

    print(
        f"  Fast scan: scored {len(scored_frames)} frames in {elapsed:.1f}s — "
        f"{'RELEVANT' if is_relevant else 'not relevant'}"
    )

    return FrameScores(
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        duration_s=duration_s,
        frames_scored=len(scored_frames),
        is_relevant=is_relevant,
        quick_answer=quick_answer,
        regions=regions,
        frame_scores=scored_frames,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Region clustering
# ──────────────────────────────────────────────────────────────────────────────


def _cluster_regions(
    frames: list[FrameScore],
    min_relevance: float = 0.2,
    gap_threshold_s: float = 10.0,
    fps: float = 30.0,
) -> list[TemporalRegion]:
    """Cluster consecutive high-scoring frames into temporal regions.

    Frames with relevance >= min_relevance are grouped into regions.
    A gap of more than gap_threshold_s seconds between high-scoring frames
    starts a new region.
    """
    if not frames:
        return []

    # Sort by timestamp
    sorted_frames = sorted(frames, key=lambda f: f.timestamp)

    regions: list[TemporalRegion] = []
    current_region_frames: list[FrameScore] = []

    for frame in sorted_frames:
        if frame.relevance_score < min_relevance:
            if current_region_frames:
                # Close current region
                regions.append(_make_region(current_region_frames))
                current_region_frames = []
            continue

        if not current_region_frames:
            current_region_frames.append(frame)
            continue

        # Check if this frame is contiguous with current region
        last = current_region_frames[-1]
        gap = frame.timestamp - last.timestamp
        if gap <= gap_threshold_s:
            current_region_frames.append(frame)
        else:
            regions.append(_make_region(current_region_frames))
            current_region_frames = [frame]

    # Close last region
    if current_region_frames:
        regions.append(_make_region(current_region_frames))

    return regions


def _make_region(frames: list[FrameScore]) -> TemporalRegion:
    assert frames  # never empty
    avg_rel = sum(f.relevance_score for f in frames) / len(frames)
    # Pick most common label across region
    label_counts: dict[str, int] = {}
    for f in frames:
        if f.top_label:
            label_counts[f.top_label] = label_counts.get(f.top_label, 0) + 1
    top_label = max(label_counts, key=lambda k: label_counts[k]) if label_counts else "object"
    return TemporalRegion(
        start_time=frames[0].timestamp,
        end_time=frames[-1].timestamp,
        avg_relevance=avg_rel,
        label=top_label,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Quick answer generation
# ──────────────────────────────────────────────────────────────────────────────


def _format_timestamp(seconds: float, fps: float = 30.0) -> str:
    """Format seconds as MM:SS."""
    total_secs = int(round(seconds))
    mins = total_secs // 60
    secs = total_secs % 60
    return f"{mins}:{secs:02d}"


def _build_quick_answer(
    frames: list[FrameScore],
    regions: list[TemporalRegion],
    query: str,
    is_relevant: bool,
    duration_s: float,
    fps: float,
) -> str:
    """Build a concise natural-language answer from scored frames."""
    if not frames:
        return "Could not read any frames from the video."

    if not is_relevant:
        return (
            f"No {query!r} detected in this video "
            f"({_format_timestamp(duration_s, fps)} total). "
            f"Consider re-phrasing the query."
        )

    total_dets = sum(f.detection_count for f in frames if f.relevance_score > 0.2)

    if not regions:
        return f"{query!r} possibly present throughout the video."

    region_strs = []
    for r in regions:
        start_str = _format_timestamp(r.start_time, fps)
        end_str = _format_timestamp(r.end_time, fps)
        region_strs.append(f"{r.label} at {start_str}–{end_str}")

    if len(region_strs) == 1:
        return (
            f"{query!r} DETECTED — {region_strs[0]}. "
            f"({len(regions[0].label)} total detections across video)"
        )

    return (
        f"{query!r} DETECTED in {len(regions)} regions: "
        f"{'; '.join(region_strs)}. "
        f"(~{total_dets} total detections)"
    )

