"""Tests for visionbrain.direction_tracking.DirectionClassifier."""

import pytest

from visionbrain.direction_tracking import DirectionClassifier


def _item(tid: int, box: list[float]) -> dict:
    return {"track_id": tid, "box": box}


def _boxes_on_line(tid: int, x_start: float, y: float, steps: int, step: float = 0.01):
    """Return a sequence of items moving along a horizontal line (+x)."""
    return [_item(tid, [x_start + i * step, y, x_start + i * step + 0.02, y + 0.02])
            for i in range(steps)]


class TestCompass:
    def test_east_motion(self):
        c = DirectionClassifier(min_move=0.005)
        items = _boxes_on_line(1, 0.1, 0.5, 4)
        out = c.classify(items)
        assert out[-1]["direction"] == "east"

    def test_west_motion(self):
        c = DirectionClassifier(min_move=0.005)
        items = _boxes_on_line(1, 0.6, 0.5, 4, step=-0.01)
        out = c.classify(items)
        assert out[-1]["direction"] == "west"

    def test_north_motion_image_coords(self):
        # y decreases => north (image y grows downward)
        c = DirectionClassifier(min_move=0.005)
        items = [_item(1, [0.3, 0.6 - i * 0.01, 0.32, 0.62 - i * 0.01]) for i in range(4)]
        out = c.classify(items)
        assert out[-1]["direction"] == "north"

    def test_south_motion_image_coords(self):
        c = DirectionClassifier(min_move=0.005)
        items = [_item(1, [0.3, 0.4 + i * 0.01, 0.32, 0.42 + i * 0.01]) for i in range(4)]
        out = c.classify(items)
        assert out[-1]["direction"] == "south"


class TestStationary:
    def test_below_min_move_is_stationary(self):
        c = DirectionClassifier(min_move=0.01)
        items = [_item(1, [0.3, 0.4, 0.32, 0.42])] * 5  # identical boxes
        out = c.classify(items)
        assert out[-1]["direction"] == "stationary"


class TestEdgeCases:
    def test_missing_track_id_passthrough(self):
        c = DirectionClassifier()
        out = c.classify([{"box": [0.1, 0.1, 0.2, 0.2]}])
        assert out[0]["direction"] == "unknown"

    def test_missing_box_passthrough(self):
        c = DirectionClassifier()
        out = c.classify([{"track_id": 1}])
        assert out[0]["direction"] == "unknown"

    def test_returns_copies_not_mutation(self):
        c = DirectionClassifier()
        original = {"track_id": 1, "box": [0.1, 0.1, 0.2, 0.2]}
        out = c.classify([original])
        assert out[0] is not original
        assert original.get("direction") is None

    def test_reset_clears_state(self):
        c = DirectionClassifier(min_move=0.005)
        out1 = c.classify(_boxes_on_line(1, 0.1, 0.5, 4))
        assert out1[-1]["direction"] == "east"
        c.reset()
        out2 = c.classify(_boxes_on_line(1, 0.1, 0.5, 1))
        # After reset, only 1 sample => not enough to classify yet.
        assert out2[-1]["direction"] == "unknown"

    def test_history_bounded(self):
        c = DirectionClassifier(min_move=0.005, max_history=3)
        # Move east for 10 steps then west; with max_history=3, the recent
        # west motion wins.
        items = _boxes_on_line(1, 0.1, 0.5, 10, step=0.01)
        items += _boxes_on_line(1, 0.25, 0.5, 10, step=-0.01)
        out = c.classify(items)
        assert out[-1]["direction"] == "west"

    def test_moved_reported(self):
        c = DirectionClassifier(min_move=0.001)
        out = c.classify(_boxes_on_line(1, 0.1, 0.5, 4, step=0.01))
        assert out[-1]["moved"] > 0.01
