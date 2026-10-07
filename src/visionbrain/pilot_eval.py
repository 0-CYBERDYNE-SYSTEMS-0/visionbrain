"""Honest-measurement pilot evaluation harness.

Replays a recorded video through the FastScan frame scorer and scores the
result against a ground-truth event label file. Produces a PilotReport that
reports raw numbers only: missed events stay missed, false alerts are
counted frame-level, and inference failures and sampling coverage are
surfaced as explicit caveats. No success framing is applied anywhere.

Ground-truth file format (JSON)::

    {
        "video": "optional-path",          # informational; --video drives the run
        "query": "person",
        "events": [
            {"label": "person", "start_s": 12.0, "end_s": 20.0, "min_count": 1}
        ]
    }

Used by the ``pilot-eval`` CLI command.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:  # lazy at runtime; frame_selector pulls in cv2
    from .frame_selector import FrameScore, FrameScores
    from PIL import Image

__all__ = ["PilotReport", "run_pilot_eval", "validate_ground_truth"]


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth validation
# ──────────────────────────────────────────────────────────────────────────────

def _is_number(value: Any) -> bool:
    """Return True only for real int/float values (bool excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_ground_truth(ground_truth: Any) -> dict:
    """Validate and normalize a parsed ground-truth JSON document.

    Args:
        ground_truth: Parsed JSON — an object with a non-empty ``"query"``
            string and an ``"events"`` list (possibly empty). Each event is an
            object with a non-empty ``"label"``, numeric ``start_s`` <=
            ``end_s``, and optional integer ``min_count`` (default 1).

    Returns:
        Normalized dict with ``"query"`` and ``"events"`` (each event normalized
        to label/start_s/end_s/min_count with float/int types).

    Raises:
        ValueError: With a specific, actionable message on any malformed field.
    """
    if not isinstance(ground_truth, dict):
        raise ValueError(
            "ground truth must be a JSON object with 'query' and 'events' keys"
        )

    query = ground_truth.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("ground truth must have a non-empty 'query' string")

    if "events" not in ground_truth:
        raise ValueError("ground truth is missing the 'events' list")
    events = ground_truth["events"]
    if not isinstance(events, list):
        raise ValueError("ground truth 'events' must be a list (it may be empty)")

    normalized: list[dict] = []
    for i, event in enumerate(events):
        where = f"events[{i}]"
        if not isinstance(event, dict):
            raise ValueError(f"{where} must be a JSON object with "
                             "label/start_s/end_s/min_count")
        label = event.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"{where} must have a non-empty 'label' string")
        start_s = event.get("start_s")
        end_s = event.get("end_s")
        if not _is_number(start_s):
            raise ValueError(f"{where} 'start_s' must be a number "
                             f"(seconds), got {start_s!r}")
        if not _is_number(end_s):
            raise ValueError(f"{where} 'end_s' must be a number "
                             f"(seconds), got {end_s!r}")
        if start_s > end_s:
            raise ValueError(f"{where}: start_s ({start_s}) must be "
                             f"<= end_s ({end_s})")
        min_count = event.get("min_count", 1)
        if not isinstance(min_count, int) or isinstance(min_count, bool) \
                or min_count < 1:
            raise ValueError(f"{where} 'min_count' must be an integer >= 1, "
                             f"got {min_count!r}")
        normalized.append({
            "label": label,
            "start_s": float(start_s),
            "end_s": float(end_s),
            "min_count": int(min_count),
        })

    return {"query": query, "events": normalized}


# ──────────────────────────────────────────────────────────────────────────────
# Injectable defaults (lazy imports — no cv2 at module import time)
# ──────────────────────────────────────────────────────────────────────────────

def _default_score_fn(
    video_path: str,
    query: str,
    *,
    sample_every_n_seconds: float,
    max_frames: int,
    resolution: int,
    min_relevance: float,
) -> "FrameScores":
    """Run the FastScan frame scorer (lazy import of frame_selector)."""
    from .frame_selector import score_frames

    return score_frames(
        video_path,
        query,
        sample_every_n_seconds=sample_every_n_seconds,
        max_frames=max_frames,
        resolution=resolution,
        min_relevance=min_relevance,
    )


def _default_frame_reader(video_path: str, frame_index: int) -> Optional["Image.Image"]:
    """Read a single video frame as an RGB PIL image (cv2, lazy import).

    Returns None when the frame cannot be read — callers must never
    fabricate a substitute image.
    """
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if not ok:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    finally:
        cap.release()


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PilotReport:
    """Honest measurement of a replayed pilot run against labeled ground truth.

    Attributes:
        video: Video path that was replayed.
        query: Query expression used for scoring.
        events_total: Number of ground-truth events.
        events_detected: Ground-truth events matched by at least one frame.
        events_missed: Ground-truth events with no matching frame.
        missed_events: Missed events as dicts (label/start_s/end_s).
        false_alert_count: Detection frames outside every event window (+- tol).
        false_alerts: The false-alert frames (timestamp, label, counts).
        per_event: Per-event dicts with detected/latency_s/support/evidence.
        coverage: Sampling coverage (duration, sampled span, scored/failed).
        runtime_s: Wall time of the scoring call in seconds.
        caveats: Honest-measurement caveats (sampling, labels, failures).
    """

    video: str
    query: str
    events_total: int
    events_detected: int
    events_missed: int
    missed_events: list[dict] = field(default_factory=list)
    false_alert_count: int = 0
    false_alerts: list[dict] = field(default_factory=list)
    per_event: list[dict] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)
    runtime_s: float = 0.0
    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the report as a JSON-safe dict."""
        return {
            "video": self.video,
            "query": self.query,
            "events_total": self.events_total,
            "events_detected": self.events_detected,
            "events_missed": self.events_missed,
            "missed_events": [dict(m) for m in self.missed_events],
            "false_alert_count": self.false_alert_count,
            "false_alerts": [dict(f) for f in self.false_alerts],
            "per_event": [dict(p) for p in self.per_event],
            "coverage": dict(self.coverage),
            "runtime_s": round(self.runtime_s, 3),
            "caveats": list(self.caveats),
        }

    def save(self, path: str | Path) -> None:
        """Write the report as JSON to ``path``."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def summary(self) -> list[str]:
        """Return human-readable summary lines with the raw numbers only."""
        lines = [
            f"Pilot evaluation: {self.video}",
            f"Query: '{self.query}'",
        ]

        line = (f"{self.events_detected}/{self.events_total} events detected, "
                f"{self.events_missed} missed; "
                f"{self.false_alert_count} false alerts")
        latencies = [p["latency_s"] for p in self.per_event
                     if p.get("detected") and p.get("latency_s") is not None]
        if latencies:
            line += (f"; per-event latency "
                     f"{min(latencies):.1f}-{max(latencies):.1f}s")
        lines.append(line)

        for m in self.missed_events:
            lines.append(
                f"  MISSED: {m['label']} @ {m['start_s']:.1f}-{m['end_s']:.1f}s"
            )

        cov = self.coverage
        cov_bits = []
        span_s = cov.get("sampled_span_s", 0.0) or 0.0
        duration_s = cov.get("duration_s", 0.0) or 0.0
        if span_s > 0 and duration_s > 0:
            cov_bits.append(f"{span_s:.0f}/{duration_s:.0f}s sampled")
        cov_bits.append(f"{cov.get('frames_scored', 0)} frames scored")
        frames_failed = cov.get("frames_failed", 0) or 0
        if frames_failed:
            cov_bits.append(f"{frames_failed} frames failed")
        lines.append("Coverage: " + ", ".join(cov_bits))
        lines.append(f"Scoring runtime: {self.runtime_s:.1f}s")

        for caveat in self.caveats:
            lines.append(f"  caveat: {caveat}")
        return lines


# ──────────────────────────────────────────────────────────────────────────────
# Core evaluation
# ──────────────────────────────────────────────────────────────────────────────

def run_pilot_eval(
    video_path: str,
    ground_truth: dict,
    *,
    score_fn: Optional[Callable[..., "FrameScores"]] = None,
    tolerance_s: float = 3.0,
    sample_every_n_seconds: float = 5.0,
    max_frames: int = 60,
    resolution: int = 360,
    min_relevance: float = 0.2,
    evidence_dir: Optional[str] = None,
    frame_reader: Optional[Callable[[str, int], Optional["Image.Image"]]] = None,
) -> PilotReport:
    """Replay a video through the frame scorer and measure against ground truth.

    The default scorer runs under host inference admission. An injected
    ``score_fn`` is used directly and remains suitable for model-free tests.

    Args:
        video_path: Path to the recorded video to replay.
        ground_truth: Parsed ground-truth JSON (see module docstring).
        score_fn: Frame scorer; defaults to frame_selector.score_frames.
            Called as ``score_fn(video_path, query, sample_every_n_seconds=...,
            max_frames=..., resolution=..., min_relevance=...)`` and must return
            a FrameScores (FrameScore entries with ``failed`` / FrameScores with
            ``frames_failed`` and ``sampled_span_s`` are honored when present).
        tolerance_s: Matching slack around each event window in seconds.
        sample_every_n_seconds: Sample one frame every N seconds.
        max_frames: Maximum number of frames to score.
        resolution: Falcon scoring resolution (lower = faster).
        min_relevance: Minimum relevance for a frame to count as a detection.
        evidence_dir: If set, save the first supporting frame of each detected
            event here as JPEG evidence.
        frame_reader: Injectable ``(video_path, frame_index) -> Optional[PIL
            image]`` used for evidence; defaults to a cv2-backed reader. When
            it returns None the event is recorded with ``evidence_saved: false``
            — no substitute image is ever fabricated.

    Returns:
        PilotReport with detected/missed events, false alerts, per-event
        latency and support, sampling coverage, runtime, and caveats.

    Raises:
        ValueError: On malformed ground truth (see validate_ground_truth).
        RuntimeError: If host inference admission is busy or unavailable.
    """
    normalized = validate_ground_truth(ground_truth)
    query = normalized["query"]
    events = normalized["events"]

    admission_handle = None
    if score_fn is None:
        from .inference_admission import InferenceAdmission

        admission = InferenceAdmission()
        admission_handle = admission.try_acquire("visionbrain-pilot-eval")
        if admission_handle is None:
            raise RuntimeError(
                f"inference admission busy: {admission.describe_holder()}"
            )
        score_fn = _default_score_fn
    if frame_reader is None:
        frame_reader = _default_frame_reader

    t_start = time.perf_counter()
    try:
        scores = score_fn(
            video_path,
            query,
            sample_every_n_seconds=sample_every_n_seconds,
            max_frames=max_frames,
            resolution=resolution,
            min_relevance=min_relevance,
        )
    finally:
        if admission_handle is not None:
            admission_handle.release()
    runtime_s = time.perf_counter() - t_start

    frame_scores: list["FrameScore"] = list(getattr(scores, "frame_scores", []) or [])
    duration_s = float(getattr(scores, "duration_s", 0.0) or 0.0)
    frames_scored = int(getattr(scores, "frames_scored", len(frame_scores)))
    frames_failed = int(getattr(scores, "frames_failed", 0) or 0)
    sampled_span_s = float(getattr(scores, "sampled_span_s", 0.0) or 0.0)

    # Detection frames: scored, not failed, at least one detection, relevant.
    detections = [
        f for f in frame_scores
        if not getattr(f, "failed", False)
        and int(f.detection_count) >= 1
        and float(f.relevance_score) >= min_relevance
    ]
    detections.sort(key=lambda f: float(f.timestamp))

    def _inside_window(t: float, event: dict) -> bool:
        return (event["start_s"] - tolerance_s) <= t <= (event["end_s"] + tolerance_s)

    # Per-event matching and evidence
    per_event: list[dict] = []
    missed_events: list[dict] = []
    events_detected = 0
    for idx, event in enumerate(events):
        support_frames = [
            f for f in detections
            if _inside_window(float(f.timestamp), event)
            and int(f.detection_count) >= event["min_count"]
        ]
        detected = bool(support_frames)
        if detected:
            events_detected += 1
        else:
            missed_events.append({
                "label": event["label"],
                "start_s": event["start_s"],
                "end_s": event["end_s"],
            })

        entry: dict = {
            "index": idx,
            "label": event["label"],
            "start_s": event["start_s"],
            "end_s": event["end_s"],
            "min_count": event["min_count"],
            "detected": detected,
            "latency_s": (float(support_frames[0].timestamp) - event["start_s"]
                          if detected else None),
            "support": [
                {
                    "frame_index": int(f.frame_index),
                    "timestamp": float(f.timestamp),
                    "relevance_score": float(f.relevance_score),
                    "top_label": f.top_label,
                    "detection_count": int(f.detection_count),
                }
                for f in support_frames
            ],
            "evidence": None,
            "evidence_saved": False,
        }

        if evidence_dir and support_frames:
            first = support_frames[0]
            out_dir = Path(evidence_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            safe_label = re.sub(r"[^\w.-]+", "_", event["label"]).strip("_") \
                or "event"
            evidence_path = out_dir / (
                f"event{idx:02d}_{safe_label}_{float(first.timestamp):.1f}s.jpg"
            )
            image = frame_reader(video_path, int(first.frame_index))
            if image is None:
                entry["evidence_saved"] = False
            else:
                image.convert("RGB").save(
                    str(evidence_path), format="JPEG", quality=90
                )
                entry["evidence"] = str(evidence_path)
                entry["evidence_saved"] = True

        per_event.append(entry)

    # False alerts: detection frames inside NO event window (+- tolerance),
    # counted frame-level so nothing is aggregated away.
    windows = [
        (event["start_s"] - tolerance_s, event["end_s"] + tolerance_s)
        for event in events
    ]
    false_alerts = [
        {
            "frame_index": int(f.frame_index),
            "timestamp": float(f.timestamp),
            "top_label": f.top_label,
            "detection_count": int(f.detection_count),
            "relevance_score": float(f.relevance_score),
        }
        for f in detections
        if not any(lo <= float(f.timestamp) <= hi for lo, hi in windows)
    ]

    caveats = [
        (f"Frames were sampled every {sample_every_n_seconds:g}s — events "
         "shorter than the sample interval can be missed entirely."),
        ("Falcon assigns labels from the query expression, so label agreement "
         "is not independent semantic confirmation; detection counts, not "
         "label semantics, are the evidence."),
    ]
    if frames_failed > 0:
        caveats.append(
            f"{frames_failed} of {frames_scored} sampled frames failed "
            "inference and were excluded from matching."
        )
    if sampled_span_s > 0 and duration_s > 0 and sampled_span_s + 1.0 < duration_s:
        caveats.append(
            f"Coverage was partial: sampled span {sampled_span_s:.1f}s of "
            f"{duration_s:.1f}s video duration."
        )

    return PilotReport(
        video=str(video_path),
        query=query,
        events_total=len(events),
        events_detected=events_detected,
        events_missed=len(missed_events),
        missed_events=missed_events,
        false_alert_count=len(false_alerts),
        false_alerts=false_alerts,
        per_event=per_event,
        coverage={
            "duration_s": duration_s,
            "sampled_span_s": sampled_span_s,
            "frames_scored": frames_scored,
            "frames_failed": frames_failed,
            "sample_every_n_seconds": sample_every_n_seconds,
            "max_frames": max_frames,
        },
        runtime_s=runtime_s,
        caveats=caveats,
    )
