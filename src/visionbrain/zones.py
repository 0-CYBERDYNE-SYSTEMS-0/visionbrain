"""Zone analytics — count objects crossing lines or inside polygons.

Built on Supervision's LineZone and PolygonZone for site-monitoring use cases:
- People or vehicles crossing a line or gate
- Objects entering/exiting a defined area
- Vehicle counting at site entrances
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import supervision as sv

from .supervision_bridge import detections_from_sam31, detections_from_falcon_boxes


# ──────────────────────────────────────────────────────────────────────────────
# Line zone counting
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LineZoneCounter:
    """Count detections crossing a line in either direction.

    Example:
        line = LineZoneCounter(start=(100, 200), end=(500, 200))
        for frame_detections in video_frames:
            counts = line.update(frame_detections)
            print(f"In: {counts['in']}, Out: {counts['out']}")
    """

    start: tuple[int, int]
    end: tuple[int, int]
    in_count: int = 0
    out_count: int = 0
    _zone: sv.LineZone = field(init=False, repr=False)
    _triggered: set[int] = field(default_factory=set, repr=False)

    def __post_init__(self):
        self._zone = sv.LineZone(
            start=sv.Point(self.start[0], self.start[1]),
            end=sv.Point(self.end[0], self.end[1]),
        )

    def update(self, detections: sv.Detections) -> dict[str, int]:
        """Process a frame of detections.

        Args:
            detections: sv.Detections with tracker_id set

        Returns:
            {"in": N, "out": N, "crossed_in": [...], "crossed_out": [...]}
        """
        if detections.tracker_id is None:
            return {"in": self.in_count, "out": self.out_count, "crossed_in": [], "crossed_out": []}

        # Trigger returns (in_crossing, out_crossing) boolean arrays
        in_crossing, out_crossing = self._zone.trigger(detections)

        crossed_in = []
        crossed_out = []
        for i, tid in enumerate(detections.tracker_id):
            if in_crossing[i] and tid not in self._triggered:
                self.in_count += 1
                self._triggered.add(tid)
                crossed_in.append(tid)
            if out_crossing[i] and tid not in self._triggered:
                self.out_count += 1
                self._triggered.add(tid)
                crossed_out.append(tid)

        return {
            "in": self.in_count,
            "out": self.out_count,
            "crossed_in": crossed_in,
            "crossed_out": crossed_out,
        }

    def reset(self) -> None:
        """Reset all counts."""
        self.in_count = 0
        self.out_count = 0
        self._triggered.clear()
        self._zone = sv.LineZone(
            start=sv.Point(self.start[0], self.start[1]),
            end=sv.Point(self.end[0], self.end[1]),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Polygon zone counting
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class PolygonZoneCounter:
    """Count detections inside a polygon region.

    Example:
        zone = PolygonZoneCounter(polygon=[(100,100), (500,100), (500,400), (100,400)])
        for frame_detections in video_frames:
            counts = zone.update(frame_detections)
            print(f"Inside: {counts['current']}, Total unique: {counts['total_unique']}")
    """

    polygon: list[tuple[int, int]]
    _zone: sv.PolygonZone = field(init=False, repr=False)
    _seen_ids: set[int] = field(default_factory=set, repr=False)
    current_count: int = 0
    total_unique: int = 0

    def __post_init__(self):
        arr = np.array(self.polygon, dtype=np.int64)
        self._zone = sv.PolygonZone(polygon=arr)

    def update(self, detections: sv.Detections) -> dict[str, int | list[int]]:
        """Process a frame of detections.

        Returns:
            {"current": N, "total_unique": N, "inside_ids": [...]}
        """
        is_inside = self._zone.trigger(detections)
        inside_ids = []

        if detections.tracker_id is not None:
            for i, tid in enumerate(detections.tracker_id):
                if is_inside[i]:
                    inside_ids.append(tid)
                    if tid not in self._seen_ids:
                        self._seen_ids.add(tid)
                        self.total_unique += 1
        else:
            # No tracker IDs — count raw detections
            inside_ids = list(range(int(is_inside.sum())))

        self.current_count = len(inside_ids)
        return {
            "current": self.current_count,
            "total_unique": self.total_unique,
            "inside_ids": inside_ids,
        }

    def reset(self) -> None:
        """Reset all counts."""
        self.current_count = 0
        self.total_unique = 0
        self._seen_ids.clear()
        arr = np.array(self.polygon, dtype=np.int64)
        self._zone = sv.PolygonZone(polygon=arr)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-zone manager
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ZoneManager:
    """Manage multiple line and polygon zones in one video stream.

    Example:
        zm = ZoneManager()
        zm.add_line("entrance", (100, 200), (500, 200))
        zm.add_polygon("yard", [(0,0), (640,0), (640,480), (0,480)])

        for frame_dets in video:
            results = zm.update_all(frame_dets)
            print(results["entrance"]["in"], results["yard"]["current"])
    """

    lines: dict[str, LineZoneCounter] = field(default_factory=dict)
    polygons: dict[str, PolygonZoneCounter] = field(default_factory=dict)

    def add_line(self, name: str, start: tuple[int, int], end: tuple[int, int]) -> None:
        """Add a named line zone."""
        self.lines[name] = LineZoneCounter(start=start, end=end)

    def add_polygon(self, name: str, polygon: list[tuple[int, int]]) -> None:
        """Add a named polygon zone."""
        self.polygons[name] = PolygonZoneCounter(polygon=polygon)

    def update_all(self, detections: sv.Detections) -> dict[str, dict]:
        """Update all zones with a frame of detections.

        Returns:
            {zone_name: zone_result_dict, ...}
        """
        results = {}
        for name, line in self.lines.items():
            results[name] = line.update(detections)
        for name, poly in self.polygons.items():
            results[name] = poly.update(detections)
        return results

    def reset_all(self) -> None:
        """Reset all zones."""
        for line in self.lines.values():
            line.reset()
        for poly in self.polygons.values():
            poly.reset()

    def summary(self) -> dict[str, dict]:
        """Get current counts for all zones without updating."""
        return {
            name: {"in": line.in_count, "out": line.out_count}
            for name, line in self.lines.items()
        } | {
            name: {"current": poly.current_count, "total_unique": poly.total_unique}
            for name, poly in self.polygons.items()
        }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: build zones from video dimensions
# ──────────────────────────────────────────────────────────────────────────────

def horizontal_line_zone(
    y_ratio: float = 0.5,
    video_width: int = 1920,
) -> LineZoneCounter:
    """Create a horizontal line zone at a given Y ratio.

    Args:
        y_ratio: 0.0 = top, 1.0 = bottom
        video_width: pixel width of the video frame
    """
    y = int(video_width * y_ratio)  # Actually this should use height... let me fix
    # Wait, the function signature is wrong. Let me provide a better one.
    raise NotImplementedError("Use LineZoneCounter directly with pixel coordinates")


# Better helper:
def line_across_frame(
    video_width: int,
    video_height: int,
    y_ratio: float = 0.5,
) -> LineZoneCounter:
    """Create a horizontal line across the full frame width.

    Args:
        video_width: frame width in pixels
        video_height: frame height in pixels
        y_ratio: 0.0 = top, 0.5 = middle, 1.0 = bottom
    """
    y = int(video_height * y_ratio)
    return LineZoneCounter(start=(0, y), end=(video_width, y))


def rectangle_zone(
    video_width: int,
    video_height: int,
    x_ratio: float = 0.25,
    y_ratio: float = 0.25,
    w_ratio: float = 0.5,
    h_ratio: float = 0.5,
) -> PolygonZoneCounter:
    """Create a rectangular polygon zone from normalized ratios.

    Args:
        video_width: frame width in pixels
        video_height: frame height in pixels
        x_ratio: left edge (0-1)
        y_ratio: top edge (0-1)
        w_ratio: width (0-1)
        h_ratio: height (0-1)
    """
    x1 = int(video_width * x_ratio)
    y1 = int(video_height * y_ratio)
    x2 = int(video_width * (x_ratio + w_ratio))
    y2 = int(video_height * (y_ratio + h_ratio))
    return PolygonZoneCounter(polygon=[
        (x1, y1), (x2, y1), (x2, y2), (x1, y2)
    ])
