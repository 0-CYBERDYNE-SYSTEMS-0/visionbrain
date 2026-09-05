"""Shared detection primitives — geometry, identity tracking, validation.

This is the single implementation of "what a detection is and how detections
relate" for every surface that shows one: the Ground Control web app, CLI
pipelines, and the live bridge hub (which imports this package directly).
Lifted from the bridge's engines.py/tracking.py after field-hardening, so the
live-tested semantics (IoU matching, ambiguity margin, cross-engine soft/hard
validation, Falcon duplicate suppression) are canonical everywhere.

All functions are pure Python with no MLX/model dependencies, so they are
safe to use and test on any host.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable

log = logging.getLogger("visionbrain.detection_core")

AGREE_IOU_THRESHOLD = 0.5
VALIDATE_MODES = ("off", "soft", "hard")

# Identity persistence must outlive engine throttle gaps (Falcon >=1s,
# LFM >=1.5s) so held re-publishes keep the same track_id/color_id instead of
# cycling the palette on every re-detect.
DEFAULT_TRACK_TTL_MS = 4000
DEFAULT_MAX_FRAME_GAP = 60
DEFAULT_MATCH_IOU = 0.30
DEFAULT_AMBIGUITY_MARGIN = 0.05
DEFAULT_PALETTE_SIZE = 8


def box_iou(a: list[float], b: list[float]) -> float:
    """IoU of two xyxy boxes in any shared coordinate space (0.0 if invalid)."""
    if len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = (float(v) for v in a[:4])
    bx1, by1, bx2, by2 = (float(v) for v in b[:4])
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def labels_compatible(a: str, b: str) -> bool:
    """Casefold equal, or one label contains the other (e.g. cow vs brown cow)."""
    la = (a or "").casefold().strip()
    lb = (b or "").casefold().strip()
    if not la or not lb:
        return False
    if la == lb:
        return True
    return la in lb or lb in la


def dedup_detections(
    items: Iterable[dict[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    """Merge near-duplicate boxes whose centers sit within ``threshold``.

    Unit-agnostic: boxes must share one coordinate space and the threshold is
    expressed in that same space (the bridge passes normalized boxes with the
    TII-official 0.01; offline pixel-space callers pass a pixel value).

    mlx-vlm does not expose Falcon's ``SamplingParams.coord_dedup_threshold``,
    so it is applied here as post-processing — the documented defense against
    Falcon's residual duplicate boxes on dense scenes. A non-positive
    threshold disables dedup.
    """
    if threshold <= 0:
        return list(items)
    kept: list[dict[str, Any]] = []
    for item in items:
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) < 4:
            kept.append(item)
            continue
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        dup = False
        for k in kept:
            kb = k.get("box")
            if not isinstance(kb, (list, tuple)) or len(kb) < 4:
                continue
            kcx = (kb[0] + kb[2]) / 2.0
            kcy = (kb[1] + kb[3]) / 2.0
            if abs(cx - kcx) <= threshold and abs(cy - kcy) <= threshold:
                dup = True
                break
        if not dup:
            kept.append(item)
    return kept


@dataclass
class _Track:
    track_id: int
    label: str
    box: list[float]
    last_frame_id: int
    last_seen_ms: int
    color_id: int


class PersistentTrackManager:
    """Assign session-local IDs without adding GPU work or unbounded history."""

    def __init__(
        self,
        *,
        ttl_ms: int = DEFAULT_TRACK_TTL_MS,
        max_frame_gap: int = DEFAULT_MAX_FRAME_GAP,
        match_iou: float = DEFAULT_MATCH_IOU,
        ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
        palette_size: int = DEFAULT_PALETTE_SIZE,
    ) -> None:
        self.ttl_ms = max(1, int(ttl_ms))
        self.max_frame_gap = max(1, int(max_frame_gap))
        self.match_iou = max(0.0, min(float(match_iou), 1.0))
        self.ambiguity_margin = max(0.0, float(ambiguity_margin))
        self.palette_size = max(1, int(palette_size))
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1
        self.epoch = 0

    def reset(self) -> int:
        """Clear all tracks; returns the new epoch counter."""
        self._tracks.clear()
        self._next_id = 1
        self.epoch += 1
        return self.epoch

    def _expire(self, frame_id: int, now_ms: int) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if now_ms - track.last_seen_ms > self.ttl_ms
            or frame_id - track.last_frame_id > self.max_frame_gap
            or frame_id < track.last_frame_id
        ]
        for track_id in expired:
            self._tracks.pop(track_id, None)

    def assign(
        self,
        items: list[dict[str, Any]],
        *,
        frame_id: int,
        now_ms: int,
    ) -> list[dict[str, Any]]:
        """Attach track_id/color_id/track_state to each item (copies returned)."""
        self._expire(frame_id, now_ms)
        used_tracks: set[int] = set()
        output: list[dict[str, Any]] = []

        for raw in items:
            item = dict(raw)
            label = str(item.get("label") or "object")
            box = list(item.get("box") or [])
            candidates: list[tuple[float, _Track]] = []
            if len(box) >= 4:
                for track in self._tracks.values():
                    if track.track_id in used_tracks:
                        continue
                    if not labels_compatible(label, track.label):
                        continue
                    iou = box_iou(box, track.box)
                    if iou >= self.match_iou:
                        candidates.append((iou, track))
            candidates.sort(key=lambda pair: pair[0], reverse=True)

            ambiguous = (
                len(candidates) > 1
                and candidates[0][0] - candidates[1][0] < self.ambiguity_margin
            )
            if candidates and not ambiguous:
                confidence, track = candidates[0]
                track.label = label
                track.box = box[:4]
                track.last_frame_id = frame_id
                track.last_seen_ms = now_ms
                state = "active"
            else:
                track_id = self._next_id
                self._next_id += 1
                track = _Track(
                    track_id=track_id,
                    label=label,
                    box=box[:4],
                    last_frame_id=frame_id,
                    last_seen_ms=now_ms,
                    color_id=(track_id - 1) % self.palette_size,
                )
                self._tracks[track_id] = track
                confidence = 0.0 if ambiguous else 1.0
                state = "uncertain" if ambiguous else "new"

            used_tracks.add(track.track_id)
            item["track_id"] = track.track_id
            item["color_id"] = track.color_id
            item["track_state"] = state
            item["track_confidence"] = round(float(confidence), 4)
            output.append(item)
        return output


def merge_validate(
    sam_items: Iterable[dict[str, Any]],
    falcon_items: Iterable[dict[str, Any]],
    mode: str = "soft",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Cross-validate two engine lists (IoU + label match).

    Returns ``(overlay_items, llm_items, stats)``.

    Modes:
      off  — all sam + falcon; llm = all
      soft — all items; also emit agree for matches; llm prefers agrees first
      hard — only agrees survive; solos suppressed (counted in stats)

    Single-engine input: validate is a no-op (no agree possible).
    """
    sam = [dict(i, source=i.get("source") or "sam") for i in sam_items]
    fal = [dict(i, source=i.get("source") or "falcon") for i in falcon_items]
    mode = mode if mode in VALIDATE_MODES else "soft"

    stats = {"sam": len(sam), "falcon": len(fal), "agree": 0, "suppressed": 0}

    if not sam or not fal:
        combined = sam + fal
        for i, it in enumerate(combined):
            # Preserve pre-assigned (e.g. SAM-tracking) ids; else stable index.
            it.setdefault("track_id", i)
        return combined, list(combined), stats

    used_f: set[int] = set()
    agrees: list[dict[str, Any]] = []
    sam_solo: list[dict[str, Any]] = []
    for s in sam:
        best_j = -1
        best_iou = 0.0
        for j, f in enumerate(fal):
            if j in used_f:
                continue
            if not labels_compatible(str(s.get("label", "")), str(f.get("label", ""))):
                continue
            iou = box_iou(list(s.get("box") or []), list(f.get("box") or []))
            if iou >= AGREE_IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used_f.add(best_j)
            f = fal[best_j]
            score_s = float(s.get("score") or 0.0)
            score_f = float(f.get("score") or 0.0)
            agree = {
                "label": s.get("label", f.get("label", "object")),
                "score": round((score_s + score_f) / 2.0, 4),
                "box": list(s.get("box") or f.get("box") or [0, 0, 0, 0]),
                "source": "agree",
            }
            if "polygon" in s:
                agree["polygon"] = s["polygon"]
            agrees.append(agree)
        else:
            sam_solo.append(s)

    fal_solo = [f for j, f in enumerate(fal) if j not in used_f]
    stats["agree"] = len(agrees)

    if mode == "off":
        overlay = sam + fal
        llm = list(overlay)
    elif mode == "hard":
        overlay = list(agrees)
        llm = list(agrees)
        stats["suppressed"] = len(sam_solo) + len(fal_solo)
        if stats["suppressed"]:
            log.debug("hard validate suppressed %d solo boxes", stats["suppressed"])
    else:  # soft
        overlay = agrees + sam_solo + fal_solo
        llm = agrees + sam_solo + fal_solo

    for i, it in enumerate(overlay):
        it.setdefault("track_id", i)
    if llm is not overlay:
        for i, it in enumerate(llm):
            it.setdefault("track_id", i)
    return overlay, llm, stats


def mask_to_polygon(
    mask: Any,
    width: int,
    height: int,
    max_points: int = 48,
) -> Optional[list[list[float]]]:
    """Extract a normalized outline polygon from a binary mask.

    Traces per-row pixel runs (leftmost/rightmost true pixel) top-down then
    back up — a dependency-free contour approximation suited to the blobby
    single-object masks SAM produces. Mask pixel coordinates are resolved
    against ``(width, height)``, so callers must pass the mask's own frame
    dimensions (SAM masks arrive frame-sized).

    Returns ``[[x, y], ...]`` normalized to 0-1 and rounded to 4 decimals, or
    ``None`` when the mask is empty/invalid or numpy is unavailable.
    """
    try:
        import numpy as np
    except ImportError:
        return None

    m = np.asarray(mask)
    if m.ndim != 2 or m.size == 0:
        return None
    m = m != 0
    true_rows = np.nonzero(m.any(axis=1))[0]
    if true_rows.size == 0:
        return None

    # Subsample rows so the emitted polygon stays bounded on wire + canvas.
    max_points = max(6, int(max_points))
    step = max(1, true_rows.size // (max_points // 2))
    sampled = true_rows[::step].tolist()
    if sampled[-1] != int(true_rows[-1]):
        sampled.append(int(true_rows[-1]))

    left: list[int] = []
    right: list[int] = []
    for r in sampled:
        cols = np.nonzero(m[r])[0]
        left.append(int(cols[0]))
        right.append(int(cols[-1]))

    poly = [[left[k], int(sampled[k])] for k in range(len(sampled))]
    poly += [[right[k], int(sampled[k])] for k in range(len(sampled) - 1, -1, -1)]

    sx = max(1.0, float(width))
    sy = max(1.0, float(height))

    def _norm(value: int, span: float) -> float:
        return round(max(0.0, min(1.0, float(value) / span)), 4)

    return [[_norm(x, sx), _norm(y, sy)] for x, y in poly]
