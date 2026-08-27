"""Per-track direction classification for live detections.

Determines motion direction per track from centroid history. This is the
"vehicles coming from one direction in one color, going another direction in
another color" primitive: given a stream of items with stable ``track_id``,
classify each track's current heading as an 8-way compass direction.

Deliberately GPU-free and dependency-free (pure Python + numpy): the bridge
consumes this via the visionbrain package, and it works in any coordinate
space as long as callers are consistent (the live bridge passes normalized
0-1 boxes; offline pipelines may pass pixels).

Design notes
------------
* Direction is derived from *displacement over time*, not instantaneous
  velocity: a track must move at least ``min_move`` (in the caller's
  coordinate units) between the oldest and newest retained centroids to be
  classified. Below that it is ``"stationary"``.
* ``history`` is bounded per track; old samples are dropped so direction is
  *recent* motion, not lifetime trajectory.
* Compass labels are in *image coordinates* (y grows downward): screen-up is
  ``"north"``, screen-right is ``"east"``, etc.

Line/gate crossing semantics (in/out counts) deliberately live in the zone
layer (see ``zones.py``) rather than here, so this primitive stays a single
responsibility: heading per track.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Compass sectors: 8-way, angles measured from +x axis, counter-clockwise.
# North = -y in image coordinates (y grows downward).
_SECTORS: list[tuple[str, float, float]] = [
    ("east", -22.5, 22.5),
    ("northeast", 22.5, 67.5),
    ("north", 67.5, 112.5),
    ("northwest", 112.5, 157.5),
    ("west", 157.5, 202.5),
    ("southwest", 202.5, 247.5),
    ("south", 247.5, 292.5),
    ("southeast", 292.5, 337.5),
]


def _compass(angle_deg: float) -> str:
    """Map a displacement angle (deg) to an 8-way compass label."""
    a = angle_deg % 360.0
    for name, lo, hi in _SECTORS:
        if lo <= a < hi:
            return name
    return "east"  # 337.5..360 wraps into east


@dataclass
class TrackMotion:
    """Motion state for a single track."""

    centroids: list[tuple[float, float]] = field(default_factory=list)
    direction: str = "unknown"
    moved: float = 0.0  # total displacement over retained history

    def push(self, cx: float, cy: float, max_history: int) -> None:
        self.centroids.append((cx, cy))
        if len(self.centroids) > max_history:
            self.centroids.pop(0)


class DirectionClassifier:
    """Classify per-track heading from a stream of detection items.

    Expected item shape (matches the bridge protocol items):
    ``{"track_id": int, "box": [x1, y1, x2, y2], ...}`` — box in any
    consistent coordinate space. Normalized 0-1 or pixels both work.
    """

    def __init__(
        self,
        *,
        min_move: float = 0.01,
        max_history: int = 10,
    ) -> None:
        self.min_move = min_move
        self.max_history = max(2, int(max_history))
        self._tracks: dict[int, TrackMotion] = {}

    def _centroid(self, box: list[float]) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _label(self, dx: float, dy: float) -> str:
        """Map a displacement to a compass label (image coords, y-down)."""
        # Image coordinates: y grows DOWNWARD. A screen-up ("north") move has
        # negative dy, but atan2(dy, dx) with a y-up convention would call that
        # "south". Flip the sign so the compass table reads in image space.
        return _compass(math.degrees(math.atan2(-dy, dx)))

    def classify(self, items: list[dict]) -> list[dict]:
        """Attach ``direction`` (+ ``moved``) to each item, in place-returning copies.

        Items without a usable ``track_id`` / ``box`` are passed through
        unchanged with ``direction="unknown"``.
        """
        out: list[dict] = []
        for item in items:
            d = dict(item)
            tid = d.get("track_id")
            box = d.get("box")
            if tid is None or not isinstance(box, (list, tuple)) or len(box) < 4:
                d["direction"] = "unknown"
                out.append(d)
                continue

            cx, cy = self._centroid(list(box[:4]))
            tm = self._tracks.setdefault(int(tid), TrackMotion())
            tm.push(cx, cy, self.max_history)

            if len(tm.centroids) >= 2:
                x0, y0 = tm.centroids[0]
                dx, dy = cx - x0, cy - y0
                moved = math.hypot(dx, dy)
                tm.moved = moved
                if moved < self.min_move:
                    tm.direction = "stationary"
                else:
                    tm.direction = self._label(dx, dy)
            else:
                tm.direction = "unknown"

            d["direction"] = tm.direction
            d["moved"] = round(tm.moved, 6)
            out.append(d)
        return out

    def reset(self) -> None:
        """Drop all track state (new scene / new shot)."""
        self._tracks.clear()
