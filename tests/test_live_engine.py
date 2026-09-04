"""Pytest suite for visionbrain.live_engine (dependency-free tests).

These tests must pass in CI with no MLX hardware and no cached weights:
live_engine defers all heavy imports (cv2, PIL, mlx, mlx_vlm) into the
worker thread body, so the module and its pure helpers are testable anywhere
fastapi is installed.

Run with:
    python -m pytest tests/test_live_engine.py -q
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

# Ensure visionbrain is importable
VBRAIN = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(VBRAIN))

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Live engine tests (no MLX / cv2 / weights required)
# ──────────────────────────────────────────────────────────────────────────────

class TestLiveEngine:
    def test_pack_frame_roundtrip(self):
        import visionbrain.live_engine as le

        jpeg = b"\xff\xd8\xff\xe0fake-jpeg-bytes\xff\xd9"
        telem = {"engine": "local", "source": "clip.mp4", "prompts": ["person"],
                 "fps": 29.97, "frame_id": 7, "resolution": "1920x1080",
                 "threshold": 0.15}
        buf = le.pack_frame(7, 123456, jpeg, telem)

        frame_id, ts_ms, jpeg_len = struct.unpack(">III", buf[:12])
        assert frame_id == 7
        assert ts_ms == 123456
        assert jpeg_len == len(jpeg)

        # JPEG bytes intact, immediately after the 12-byte header.
        assert buf[12:12 + jpeg_len] == jpeg

        # telem_len comes AFTER the jpeg, then the JSON bytes.
        (telem_len,) = struct.unpack(">I", buf[12 + jpeg_len:16 + jpeg_len])
        assert telem_len == len(buf) - (16 + jpeg_len)
        assert json.loads(buf[16 + jpeg_len:].decode("utf-8")) == telem

        # Total frame length is exactly header + jpeg + telem header + telem.
        assert len(buf) == 12 + jpeg_len + 4 + telem_len

    def test_pack_frame_u32_wrapping(self):
        import visionbrain.live_engine as le

        # Large frame_id / epoch-ms timestamps must mask into u32, not raise.
        buf = le.pack_frame(2**32 + 5, 1_750_000_000_000, b"j", {})
        frame_id, ts_ms, jpeg_len = struct.unpack(">III", buf[:12])
        assert frame_id == 5
        assert jpeg_len == 1
        assert 0 <= ts_ms <= 0xFFFFFFFF

    def test_make_item_normalization(self):
        import visionbrain.live_engine as le

        item = le.make_item([80, 70, 100, 100], 100, 100, "car", 0.91234, 3, "east")
        assert item["box"] == [0.8, 0.7, 1.0, 1.0]
        assert item["label"] == "car"
        assert item["score"] == 0.912
        assert item["track_id"] == 3
        assert item["color_id"] == 3  # mirrors track_id for stable palette
        assert item["direction"] == "east"

    def test_make_item_edge_box(self):
        import visionbrain.live_engine as le

        # Box flush against the right/bottom edge stays within 0-1.
        item = le.make_item([640, 360, 640, 360], 640, 360, "person", 0.5, 1, "south")
        assert item["box"] == [1.0, 1.0, 1.0, 1.0]

        # Origin corner normalizes to zero.
        item = le.make_item([0, 0, 32, 16], 640, 360, "truck", 0.75, 2, "north")
        assert item["box"] == [0.0, 0.0, 0.05, round(16 / 360, 4)]

        # Out-of-range pixel values clamp into 0-1.
        item = le.make_item([-10, -10, 9999, 9999], 100, 100, "x", 1.0, 0, "east")
        assert item["box"] == [0.0, 0.0, 1.0, 1.0]

    def test_validate_control_file_start(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "action": "start", "source": "file",
            "file_id": "abc123", "prompts": ["person", "car"],
        })
        assert action == "start"
        assert payload["source"] == "file"
        assert payload["file_id"] == "abc123"
        assert payload["prompts"] == ["person", "car"]

    def test_validate_control_webcam_start(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "action": "start", "source": "webcam",
            "camera": 0, "prompts": ["truck"],
        })
        assert action == "start"
        assert payload["source"] == "webcam"
        assert payload["camera"] == 0

    def test_validate_control_missing_file_id(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "action": "start", "source": "file", "prompts": ["person"],
        })
        assert action == "unknown"
        assert payload == {}

        # Empty string and non-string file_id are rejected too.
        for bad in ("", "   ", 7, None):
            action, _ = le.validate_control({
                "action": "start", "source": "file",
                "file_id": bad, "prompts": ["person"],
            })
            assert action == "unknown"

    def test_validate_control_bad_source(self):
        import visionbrain.live_engine as le

        for bad in ("rtsp", "http", "", None):
            action, payload = le.validate_control({
                "action": "start", "source": bad,
                "file_id": "abc", "prompts": ["person"],
            })
            assert action == "unknown"
            assert payload == {}

    def test_validate_control_bad_webcam(self):
        import visionbrain.live_engine as le

        # Negative camera, missing camera, and bools are rejected.
        assert le.validate_control({
            "action": "start", "source": "webcam",
            "camera": -1, "prompts": ["car"],
        })[0] == "unknown"
        assert le.validate_control({
            "action": "start", "source": "webcam", "prompts": ["car"],
        })[0] == "unknown"
        assert le.validate_control({
            "action": "start", "source": "webcam",
            "camera": True, "prompts": ["car"],
        })[0] == "unknown"

    def test_validate_control_empty_prompts(self):
        import visionbrain.live_engine as le

        # Empty list, non-list, and blank/None entries are all invalid —
        # on both start and set_prompts.
        bad_prompts = ([], "person", ["", " "], [None], [123], None)
        for prompts in bad_prompts:
            action, _ = le.validate_control({
                "action": "start", "source": "file",
                "file_id": "abc", "prompts": prompts,
            })
            assert action == "unknown"
            action, _ = le.validate_control({
                "action": "set_prompts", "prompts": prompts,
            })
            assert action == "unknown"

    def test_validate_control_set_prompts_ok(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "action": "set_prompts", "prompts": ["truck", "bus"],
        })
        assert action == "set_prompts"
        assert payload == {"prompts": ["truck", "bus"]}

    def test_validate_control_stop_and_shutdown(self):
        import visionbrain.live_engine as le

        assert le.validate_control({"action": "stop"}) == ("stop", {})
        assert le.validate_control({"action": "shutdown"}) == ("shutdown", {})

    def test_validate_control_garbage(self):
        import visionbrain.live_engine as le

        # No exceptions on garbage — always ("unknown", {}).
        for garbage in (
            {}, {"action": "explode"}, {"action": None},
            {"action": 123}, {"action": "start"},  # missing everything else
            "start", 42, None, ["/api/live/ws"], {"Action": "stop"},
        ):
            action, payload = le.validate_control(garbage)
            assert action == "unknown"
            assert payload == {}

    def test_module_import_lazy(self):
        """Importing the module must succeed with no mlx/weights present."""
        import visionbrain.live_engine as le

        assert hasattr(le, "router")
        assert callable(le.pack_frame)
        assert callable(le.make_item)
        assert callable(le.validate_control)
        assert callable(le.configure)
        assert callable(le.resolve_file_id)
        assert callable(le.live_ws)

    def test_resolve_file_id_guards(self, tmp_path, monkeypatch):
        import visionbrain.live_engine as le

        # Baseline: record the current global so monkeypatch restores it.
        monkeypatch.setattr(le, "_uploads_dir", le._uploads_dir)

        # Unconfigured → never resolves.
        le._uploads_dir = None
        assert le.resolve_file_id("abc") is None

        le.configure(tmp_path)
        (tmp_path / "abc123.mp4").write_bytes(b"x")

        # Happy path: single glob match resolves to the file.
        got = le.resolve_file_id("abc123")
        assert got is not None
        assert got.name == "abc123.mp4"

        # Ambiguous prefix (two matches) → rejected.
        (tmp_path / "abc456.mp4").write_bytes(b"x")
        assert le.resolve_file_id("abc") is None
        # The specific id still resolves.
        assert le.resolve_file_id("abc123") is not None

        # Traversal attempts are rejected outright.
        assert le.resolve_file_id("../abc123") is None
        assert le.resolve_file_id("a/b") is None
        assert le.resolve_file_id("a\\b") is None
        assert le.resolve_file_id("") is None

        # No match at all.
        assert le.resolve_file_id("zzz") is None
