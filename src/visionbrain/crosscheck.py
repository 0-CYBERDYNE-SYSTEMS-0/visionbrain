"""Cross-engine detection validation — SAM 3.1 vs Falcon Perception.

When Falcon Perception refines key frames in the ``analyze`` pipeline, this
module validates Falcon's boxes against SAM's tracked detections at the same
frame indices: greedy highest-IoU-first one-to-one matching, per-frame
agreement scoring, and an aggregate summary for the ops log and the Gemma
reasoning context.

Pure Python (stdlib + :mod:`visionbrain.detection_core` only) — no MLX, safe
to import and test on any host.

Agreement metric: ``matched / max(1, max(len(sam), len(falcon)))`` — the
fraction of the larger engine's detection set that both engines agree on.
A single engine with no counterpart scores 0.0; two empty lists score 0.0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .detection_core import box_iou

DEFAULT_IOU_THRESHOLD = 0.5


@dataclass
class CrosscheckResult:
    """Outcome of validating one frame's Falcon detections against SAM's.

    ``frame_index`` defaults to ``-1`` (unknown at match time); callers that
    know the frame set it on the returned instance.
    """

    frame_index: int = -1
    matched: int = 0
    sam_only: int = 0
    falcon_only: int = 0
    agreement: float = 0.0
    matches: list[dict[str, Any]] = field(default_factory=list)


def _get_box(det: dict[str, Any]) -> list[float]:
    """Return the xyxy box of a detection dict (bbox_xyxy preferred, box fallback)."""
    box = det.get("bbox_xyxy") if det.get("bbox_xyxy") is not None else det.get("box")
    if not isinstance(box, (list, tuple)):
        return []
    return [float(v) for v in box[:4]]


def falcon_to_dets(
    detection_results: Iterable[Any],
    orig_w: int,
    orig_h: int,
) -> list[dict[str, Any]]:
    """Convert Falcon ``DetectionResult`` items to pixel-space detection dicts.

    Falcon reports normalized center/size (``cx``, ``cy``, ``h``, ``w``);
    SAM reports pixel ``bbox_xyxy``. Accepts dataclass-like objects or dicts
    with those keys. Corners are ``(cx ± w/2, cy ± h/2) * (orig_w, orig_h)``.
    """
    w = float(orig_w)
    h = float(orig_h)
    dets: list[dict[str, Any]] = []
    for r in detection_results:
        if isinstance(r, dict):
            cx = float(r.get("cx", 0.0))
            cy = float(r.get("cy", 0.0))
            bh = float(r.get("h", 0.0))
            bw = float(r.get("w", 0.0))
            label = str(r.get("label", "object"))
            score = float(r.get("score", 0.0))
        else:
            cx = float(getattr(r, "cx", 0.0))
            cy = float(getattr(r, "cy", 0.0))
            bh = float(getattr(r, "h", 0.0))
            bw = float(getattr(r, "w", 0.0))
            label = str(getattr(r, "label", "object"))
            score = float(getattr(r, "score", 0.0))
        x1 = (cx - bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        x2 = (cx + bw / 2.0) * w
        y2 = (cy + bh / 2.0) * h
        dets.append({
            "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "label": label,
            "score": score,
            "source": "falcon",
        })
    return dets


def crosscheck(
    sam_dets: list[dict[str, Any]],
    falcon_dets: list[dict[str, Any]],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> CrosscheckResult:
    """Match SAM detections against Falcon detections for one frame.

    Greedy highest-IoU-first one-to-one matching using
    :func:`visionbrain.detection_core.box_iou`: each SAM det matches at most
    one Falcon det and vice versa; a pair matches when IoU >= ``iou_threshold``
    (labels are reported per pair but not required to be compatible, so
    cross-engine label disagreement stays visible).

    Match dicts carry ``sam_label``, ``falcon_label``, ``iou`` and
    ``sam_track_id`` (``None`` when the SAM det has no track id). The returned
    ``CrosscheckResult.frame_index`` is ``-1``; set it if you know the frame.
    """
    sam = list(sam_dets or [])
    fal = list(falcon_dets or [])

    pairs: list[tuple[float, int, int]] = []
    for i, s in enumerate(sam):
        sbox = _get_box(s)
        for j, f in enumerate(fal):
            iou = box_iou(sbox, _get_box(f))
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_s: set[int] = set()
    used_f: set[int] = set()
    matches: list[dict[str, Any]] = []
    for iou, i, j in pairs:
        if i in used_s or j in used_f:
            continue
        used_s.add(i)
        used_f.add(j)
        matches.append({
            "sam_label": sam[i].get("label", ""),
            "falcon_label": fal[j].get("label", ""),
            "iou": round(float(iou), 4),
            "sam_track_id": sam[i].get("track_id"),
        })

    matched = len(matches)
    agreement = matched / max(1, max(len(sam), len(fal)))
    return CrosscheckResult(
        frame_index=-1,
        matched=matched,
        sam_only=len(sam) - matched,
        falcon_only=len(fal) - matched,
        agreement=agreement,
        matches=matches,
    )


def summarize(results: list[CrosscheckResult]) -> dict[str, Any]:
    """Aggregate per-frame results: totals plus mean per-frame agreement.

    ``agreement`` is the mean of the per-frame agreements (0.0 when no frames).
    """
    frames = len(results)
    matched = sum(r.matched for r in results)
    sam_only = sum(r.sam_only for r in results)
    falcon_only = sum(r.falcon_only for r in results)
    agreement = (sum(r.agreement for r in results) / frames) if frames else 0.0
    return {
        "frames": frames,
        "matched": matched,
        "sam_only": sam_only,
        "falcon_only": falcon_only,
        "agreement": agreement,
    }
