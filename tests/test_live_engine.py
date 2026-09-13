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
import threading
import time
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

        # An EMPTY list is valid on start and set_prompts — the hub-protocol
        # "detection off" pause semantics the Android clients send (the server
        # stops until prompts are restored). Non-list and blank/non-string
        # entries stay invalid.
        action, payload = le.validate_control({
            "action": "start", "source": "file",
            "file_id": "abc", "prompts": [],
        })
        assert (action, payload) == (
            "start", {"source": "file", "file_id": "abc", "prompts": []},
        )
        assert le.validate_control({"action": "set_prompts", "prompts": []}) == (
            "set_prompts", {"prompts": []},
        )
        bad_prompts = ("person", ["", " "], [None], [123], None)
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
        # Smart-capture helpers must be importable without heavy deps too.
        assert callable(le.validate_zones)
        assert callable(le.sanitize_clip_name)
        assert callable(le.validate_stream_url)
        assert callable(le.redact_url)
        assert callable(le.RectZone)
        assert callable(le.DwellTracker)
        assert callable(le.DirectionTriggerState)

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

    def test_configure_clips_dir(self, tmp_path, monkeypatch):
        import visionbrain.live_engine as le

        monkeypatch.setattr(le, "_uploads_dir", None)
        monkeypatch.setattr(le, "_clips_dir", None)

        clips = tmp_path / "nested" / "clips"
        le.configure(tmp_path, clips)
        assert le._clips_dir == clips
        assert clips.is_dir()  # created (with parents) when missing

        # clips_dir is optional — None disables capture but keeps uploads.
        le.configure(tmp_path)
        assert le._uploads_dir == tmp_path
        assert le._clips_dir is None

        # An uncreatable clips dir degrades gracefully to disabled capture.
        blocked = tmp_path / "f"
        blocked.write_bytes(b"not a dir")
        le.configure(tmp_path, blocked / "clips")
        assert le._clips_dir is None


# ──────────────────────────────────────────────────────────────────────────────
# Network-stream sources: validate_stream_url / redact_url / url starts
# ──────────────────────────────────────────────────────────────────────────────

class TestStreamUrl:
    def test_validate_stream_url_accepts(self):
        import visionbrain.live_engine as le

        for url in (
            "rtsp://cam.local:554/stream",
            "rtsps://cam.local:322/live",
            "http://cam.local:8080/video",
            "https://example.com/live.m3u8",
            "RTSP://CAM:554/x",              # scheme match is case-insensitive
            "RtSpS://cam/x",
            "HTTP://cam/v",
            "HTTPS://example.com/a",
            "rtsp://user:pass@cam:554/s",    # userinfo allowed (redacted later)
            "https://a.b",                   # minimal non-empty remainder
        ):
            assert le.validate_stream_url(url) is True, url

    def test_validate_stream_url_rejects(self):
        import visionbrain.live_engine as le

        for url in (
            "",                              # empty
            "   ",                           # blank — no scheme
            "camera.local:554/stream",       # plain host, no scheme
            "file:///x.mp4",                 # disallowed scheme
            "ftp://x",                       # disallowed scheme
            "gopher://host/x",               # disallowed scheme
            "rtsp://",                       # empty remainder after scheme
            "http://", "https://", "rtsps://",
            "rtsp://cam host/stream",        # whitespace in remainder
            "rtsp://cam\tstream",            # tab counts as whitespace
            "https://example.com/a b?c=d",   # space in query
            " rtsp://cam",                   # leading space breaks the scheme
            "rtsp://cam/path ",              # trailing space in remainder
            "rtsp://" + "a" * 494,           # 501 chars total — too long
            "x" * 501,                       # way too long, no scheme anyway
            None, 42, True, b"rtsp://cam", ["rtsp://cam"],  # non-strings
        ):
            assert le.validate_stream_url(url) is False, repr(url)

    def test_validate_stream_url_length_boundary(self):
        import visionbrain.live_engine as le

        # "rtsp://" is 7 chars: 493 fill = exactly 500 (ok), 494 = 501 (no).
        assert le.validate_stream_url("rtsp://" + "a" * 493) is True
        assert le.validate_stream_url("rtsp://" + "a" * 494) is False

    def test_redact_url_masks_credentials(self):
        import visionbrain.live_engine as le

        assert (
            le.redact_url("rtsp://admin:secret@cam:554/s")
            == "rtsp://admin:***@cam:554/s"
        )
        # Empty user → the whole userinfo collapses to ***@.
        assert le.redact_url("rtsp://:s3cret@cam:554/s") == "rtsp://***@cam:554/s"
        # Works on http(s) too, and for later path segments.
        assert (
            le.redact_url("https://bob:hunter2@example.com/live?a=1")
            == "https://bob:***@example.com/live?a=1"
        )

    def test_redact_url_noop_without_userinfo(self):
        import visionbrain.live_engine as le

        # No @ in the authority → unchanged; port colons must not confuse it.
        assert le.redact_url("rtsp://cam:554/live") == "rtsp://cam:554/live"
        assert le.redact_url("http://example.com/a?b=1") == "http://example.com/a?b=1"
        # An @ appearing later (path/query) is not userinfo.
        assert (
            le.redact_url("rtsp://cam:554/a?user=me@x")
            == "rtsp://cam:554/a?user=me@x"
        )

    def test_validate_control_url_start(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "type": "start", "source": "url",
            "url": "rtsp://admin:secret@cam:554/stream", "prompts": ["person"],
        })
        assert action == "start"
        assert payload["source"] == "url"
        assert payload["url"] == "rtsp://admin:secret@cam:554/stream"
        assert payload["prompts"] == ["person"]

        # The "action" alias works for url starts too, with tuning knobs.
        action, payload = le.validate_control({
            "action": "start", "source": "url",
            "url": "https://cam.local/live.m3u8", "prompts": ["car"],
            "threshold": 0.3,
        })
        assert action == "start"
        assert payload["url"] == "https://cam.local/live.m3u8"
        assert payload["threshold"] == 0.3

    def test_validate_control_url_start_invalid(self):
        import visionbrain.live_engine as le

        def start(url):
            return le.validate_control({
                "type": "start", "source": "url", "url": url, "prompts": ["x"],
            })

        # Missing url key.
        action, payload = le.validate_control({
            "type": "start", "source": "url", "prompts": ["x"],
        })
        assert action == "unknown"
        assert payload == {}

        # Bad schemes / malformed urls / empty.
        for bad in (
            "", "   ", "camera.local:554/stream", "file:///x.mp4", "ftp://x",
            "rtsp://", "rtsp://a b/c", "rtsp://" + "a" * 494,
        ):
            action, payload = start(bad)
            assert action == "unknown", bad
            assert payload == {}

        # Non-string urls.
        for bad in (None, 42, True, ["rtsp://cam"], {"url": "rtsp://cam"}):
            action, payload = start(bad)
            assert action == "unknown", bad
            assert payload == {}

        # Unknown sources stay unknown (no collision with the new branch).
        for bad in ("rtsp", "http", "url ", "URL"):
            action, _ = le.validate_control({
                "action": "start", "source": bad, "prompts": ["x"],
            })
            assert action == "unknown"




class TestSanitizeClipName:
    def test_accepts_safe_names(self):
        import visionbrain.live_engine as le

        assert le.sanitize_clip_name("clip_1_line_cross.mp4") == "clip_1_line_cross.mp4"
        assert le.sanitize_clip_name("A9-_.mp4") == "A9-_.mp4"

    def test_rejects_unsafe_names(self):
        import visionbrain.live_engine as le

        # Path traversal, separators, wrong extension, non-strings.
        for bad in ("../evil", "a/b.mp4", "a\\b.mp4", "x.txt", "..",
                    "", ".mp4.mp3", None, 42, b"clip.mp4", "clip mp4"):
            assert le.sanitize_clip_name(bad) is None, bad


class TestValidateZones:
    def test_valid_line_and_rect_normalized(self):
        import visionbrain.live_engine as le

        out = le.validate_zones([
            {"kind": "line", "name": "gate", "a": [0, 0.5], "b": [1, 0.5]},
            {"kind": "rect", "x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.9},
        ])
        assert out[0] == {"kind": "line", "name": "gate", "a": [0.0, 0.5], "b": [1.0, 0.5]}
        # Missing/None name defaults to "zone N" (1-based position).
        assert out[1]["name"] == "zone 2"
        assert out[1]["x1"] == 0.1 and out[1]["y2"] == 0.9

    def test_rejects_bad_zones(self):
        import visionbrain.live_engine as le

        bad_sets = [
            "nope",                                   # not a list
            [{"kind": "poly", "a": [0, 0], "b": [1, 1]}],   # bad kind
            [{"kind": "line", "a": [0, 0]}],                 # missing b
            [{"kind": "line", "a": [0, 0], "b": [1]}],       # 1-number pair
            [{"kind": "line", "a": [0, 0], "b": [1, 1.5]}],  # out of 0-1
            [{"kind": "line", "a": [0, 0], "b": [1, True]}], # bool coords
            [{"kind": "line", "a": [0, 0], "b": [1, "x"]}],  # non-number
            [{"kind": "rect", "x1": 0.5, "y1": 0, "x2": 0.5, "y2": 1}],  # x1 == x2
            [{"kind": "rect", "x1": 0.9, "y1": 0, "x2": 0.1, "y2": 1}],  # x1 > x2
            [{"kind": "rect", "y1": 0.9, "x1": 0, "x2": 1, "y2": 0.1}],  # y1 > y2
            [{"kind": "rect", "x1": 0, "y1": 0, "x2": 1.2, "y2": 1}],    # out of range
            [{"kind": "rect", "x1": 0, "y1": 0, "x2": 1, "y2": 1, "name": 7}],  # bad name
            [{"kind": "line", "a": [0, 0], "b": [1, 1], "name": "x" * 41}],      # name too long
            [{"kind": "line", "a": [0, 0], "b": [1, 1]}] * 9,                    # > 8 zones
        ]
        for zones in bad_sets:
            with pytest.raises(ValueError):
                le.validate_zones(zones)

    def test_max_eight_zones_ok_and_names_stripped(self):
        import visionbrain.live_engine as le

        eight = [{"kind": "line", "a": [0, 0], "b": [1, 1]} for _ in range(8)]
        out = le.validate_zones(eight)
        assert len(out) == 8
        assert [z["name"] for z in out] == [f"zone {i}" for i in range(1, 9)]
        assert le.validate_zones([
            {"kind": "line", "a": [0, 0], "b": [1, 1], "name": "  gate  "}
        ])[0]["name"] == "gate"


class TestValidateBox:
    def test_valid_boxes(self):
        import visionbrain.live_engine as le

        assert le.validate_box([0.1, 0.2, 0.3, 0.4]) == [0.1, 0.2, 0.3, 0.4]
        # Tuples and int coordinates are accepted and floatified.
        assert le.validate_box((0, 0, 1, 1)) == [0.0, 0.0, 1.0, 1.0]
        assert le.validate_box([0, 0.5, 0.25, 1]) == [0.0, 0.5, 0.25, 1.0]

    def test_rejects_inverted_boxes(self):
        import visionbrain.live_engine as le

        # x1 >= x2 or y1 >= y2 — consistent with rect zone ordering.
        assert le.validate_box([0.5, 0.0, 0.5, 1.0]) is None   # x1 == x2
        assert le.validate_box([0.9, 0.0, 0.1, 1.0]) is None   # x1 > x2
        assert le.validate_box([0.0, 0.9, 1.0, 0.1]) is None   # y1 > y2
        assert le.validate_box([0.0, 0.5, 1.0, 0.5]) is None   # y1 == y2

    def test_rejects_out_of_range(self):
        import visionbrain.live_engine as le

        assert le.validate_box([-0.1, 0, 0.5, 0.5]) is None
        assert le.validate_box([0.0, 0.0, 1.0, 1.2]) is None
        assert le.validate_box([0.0, -1, 1.0, 1]) is None

    def test_rejects_bools(self):
        import visionbrain.live_engine as le

        # bool is an int subclass — must be rejected explicitly.
        assert le.validate_box([True, 0, 1, 1]) is None
        assert le.validate_box([0, 0, False, 1]) is None

    def test_rejects_wrong_arity_and_types(self):
        import visionbrain.live_engine as le

        for bad in (
            [0.0, 0.0, 1.0],            # 3 numbers
            [0.0, 0.0, 1.0, 1.0, 0.5],  # 5 numbers
            [],                          # empty
            "0,0,1,1",                   # string
            {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
            None, 42,
            [0.0, 0.0, "1", 1.0],       # non-number entry
            [0.0, 0.0, None, 1.0],
        ):
            assert le.validate_box(bad) is None, bad


class TestTargetLabelAt:
    def test_containment_first_match_wins(self):
        import visionbrain.live_engine as le

        targets = [
            {"box": [0.0, 0.0, 0.5, 0.5], "label": "target 1"},
            {"box": [0.25, 0.25, 0.75, 0.75], "label": "target 2"},
        ]
        assert le.target_label_at(0.1, 0.1, targets) == "target 1"
        # Overlap region resolves to the FIRST matching target.
        assert le.target_label_at(0.3, 0.3, targets) == "target 1"
        assert le.target_label_at(0.6, 0.6, targets) == "target 2"
        assert le.target_label_at(0.9, 0.9, targets) is None

    def test_edges_inclusive_and_malformed_targets(self):
        import visionbrain.live_engine as le

        targets = [{"box": [0.1, 0.2, 0.3, 0.4], "label": "target 7"}]
        # Edges are inclusive (mirrors RectZone.contains).
        assert le.target_label_at(0.1, 0.2, targets) == "target 7"
        assert le.target_label_at(0.3, 0.4, targets) == "target 7"
        assert le.target_label_at(0.09, 0.2, targets) is None
        assert le.target_label_at(0.1, 0.41, targets) is None
        # Malformed target entries are skipped, never raised on.
        assert le.target_label_at(0.2, 0.3, [{"nope": 1}, None, {}]) is None


class TestValidateTargetControls:
    def test_add_prompt_box_valid(self, monkeypatch):
        import visionbrain.live_engine as le

        monkeypatch.setattr(le, "_pending_targets", None)
        action, payload = le.validate_control({
            "type": "add_prompt_box", "box": [0.1, 0.2, 0.3, 0.4],
        })
        assert action == "add_prompt_box"
        assert payload == {"box": [0.1, 0.2, 0.3, 0.4], "label": None}

        # "action" alias, explicit label (stripped), int coords floatified.
        action, payload = le.validate_control({
            "action": "add_prompt_box", "box": [0, 0, 1, 1], "label": "  gate  ",
        })
        assert action == "add_prompt_box"
        assert payload == {"box": [0.0, 0.0, 1.0, 1.0], "label": "gate"}

    def test_add_prompt_box_missing_or_bad_box(self, monkeypatch):
        import visionbrain.live_engine as le

        monkeypatch.setattr(le, "_pending_targets", None)
        bad_boxes = (
            None,                       # missing
            "nope", [0, 0, 1],          # wrong type / arity
            [0.9, 0, 0.1, 1],           # inverted
            [0, 0, 1.2, 1],             # out of range
            [True, 0, 1, 1],            # bool coordinate
        )
        for box in bad_boxes:
            action, payload = le.validate_control({
                "type": "add_prompt_box", "box": box,
            })
            assert action == "unknown", box
            assert payload == {}

    def test_add_prompt_box_label_rules(self, monkeypatch):
        import visionbrain.live_engine as le

        monkeypatch.setattr(le, "_pending_targets", None)
        # >40 chars is rejected; non-strings are rejected.
        action, _ = le.validate_control({
            "type": "add_prompt_box", "box": [0, 0, 1, 1], "label": "x" * 41,
        })
        assert action == "unknown"
        action, _ = le.validate_control({
            "type": "add_prompt_box", "box": [0, 0, 1, 1], "label": 7,
        })
        assert action == "unknown"
        # Empty/whitespace labels fall back to the engine-side default.
        for blank in ("", "   ", None):
            action, payload = le.validate_control({
                "type": "add_prompt_box", "box": [0, 0, 1, 1], "label": blank,
            })
            assert action == "add_prompt_box"
            assert payload["label"] is None

    def test_add_prompt_box_pending_cap(self, monkeypatch):
        import visionbrain.live_engine as le

        eight = [{"box": [0.0, 0.0, 0.1, 0.1], "label": None}] * 8
        monkeypatch.setattr(le, "_pending_targets", list(eight))
        action, payload = le.validate_control({
            "type": "add_prompt_box", "box": [0, 0, 1, 1],
        })
        assert action == "unknown"
        assert payload == {}

        # Seven staged targets still leave room for one more.
        monkeypatch.setattr(le, "_pending_targets", list(eight[:7]))
        assert le.validate_control({
            "type": "add_prompt_box", "box": [0, 0, 1, 1],
        })[0] == "add_prompt_box"

    def test_remove_targets(self):
        import visionbrain.live_engine as le

        assert le.validate_control({"type": "remove_targets"}) == ("remove_targets", {})
        assert le.validate_control({"action": "remove_targets"}) == ("remove_targets", {})


class TestWorkerTargets:
    """Box-target state on the worker — pure (no heavy imports in __init__)."""

    @staticmethod
    def _worker():
        import visionbrain.live_engine as le

        return le._EngineWorker(
            {"source": "webcam", "camera": 0, "prompts": ["person"]}, set()
        )

    def test_add_target_default_labels_sequence(self):
        worker = self._worker()
        assert worker.add_target([0, 0, 0.5, 0.5]) == 1
        assert worker.add_target([0.1, 0.1, 0.6, 0.6]) == 2
        assert worker.get_targets() == [
            {"box": [0.0, 0.0, 0.5, 0.5], "label": "target 1"},
            {"box": [0.1, 0.1, 0.6, 0.6], "label": "target 2"},
        ]
        # An explicit label wins over the default.
        assert worker.add_target([0, 0, 1, 1], "gate") == 3
        assert worker.get_targets()[2]["label"] == "gate"

    def test_add_target_caps_at_eight(self):
        worker = self._worker()
        for i in range(8):
            assert worker.add_target([0, 0, 0.1, 0.1]) == i + 1
        assert worker.add_target([0, 0, 0.1, 0.1]) is None
        assert len(worker.get_targets()) == 8

    def test_clear_targets_keeps_sequence(self):
        worker = self._worker()
        worker.add_target([0, 0, 0.5, 0.5])
        worker.clear_targets()
        assert worker.get_targets() == []
        # The per-engine sequence keeps running after a clear.
        assert worker.add_target([0, 0, 0.5, 0.5]) == 2
        assert worker.get_targets()[0]["label"] == "target 2"

    def test_get_targets_returns_copies(self):
        worker = self._worker()
        worker.add_target([0, 0, 0.5, 0.5])
        snapshot = worker.get_targets()
        snapshot[0]["box"][0] = 9.9
        snapshot[0]["label"] = "mutated"
        assert worker.get_targets()[0]["box"][0] == 0.0
        assert worker.get_targets()[0]["label"] == "target 1"


class TestValidateControlSmartCapture:
    def test_set_zones_valid(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "type": "set_zones",
            "zones": [
                {"kind": "line", "name": "gate", "a": [0, 0.5], "b": [1, 0.5]},
                {"kind": "rect", "name": "yard", "x1": 0, "y1": 0, "x2": 0.5, "y2": 0.5},
            ],
        })
        assert action == "set_zones"
        assert [z["name"] for z in payload["zones"]] == ["gate", "yard"]

        # Empty set is valid (clears all zones); "action" alias works too.
        assert le.validate_control({"type": "set_zones", "zones": []})[1] == {"zones": []}
        action, _ = le.validate_control({"action": "set_zones", "zones": []})
        assert action == "set_zones"

    def test_set_zones_invalid(self):
        import visionbrain.live_engine as le

        good_line = {"kind": "line", "a": [0, 0], "b": [1, 1]}
        bad_sets = [
            None, "x", 7,                     # zones not a list
            [{"kind": "poly"}],               # bad kind
            [{"kind": "line", "a": [0, 0], "b": [2, 2]}],   # coords out of 0-1
            [{"kind": "rect", "x1": 0.9, "y1": 0, "x2": 0.1, "y2": 1}],  # x1 > x2
            [good_line] * 9,                  # > 8 zones
        ]
        for zones in bad_sets:
            action, payload = le.validate_control({"type": "set_zones", "zones": zones})
            assert action == "unknown", zones
            assert payload == {}

    def test_set_triggers_valid(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "type": "set_triggers", "line_cross": True, "direction": "north",
            "dwell_s": 5, "clip": False, "pre_s": 3, "post_s": 2.5,
        })
        assert action == "set_triggers"
        assert payload == {
            "line_cross": True, "direction": "north", "dwell_s": 5.0,
            "clip": False, "pre_s": 3.0, "post_s": 2.5,
        }

        # All keys optional — a partial update carries only what was sent.
        action, payload = le.validate_control({"action": "set_triggers", "dwell_s": 10})
        assert action == "set_triggers"
        assert payload == {"dwell_s": 10.0}

        # Every compass direction is accepted; zero dwell/edge values are fine.
        for d in ("none", "any", "north", "northeast", "east", "southeast",
                  "south", "southwest", "west", "northwest"):
            assert le.validate_control({"type": "set_triggers", "direction": d})[0] == "set_triggers"
        assert le.validate_control({"type": "set_triggers", "dwell_s": 0})[1] == {"dwell_s": 0.0}

    def test_set_triggers_invalid(self):
        import visionbrain.live_engine as le

        bad_msgs = [
            {"direction": "up"},              # not a compass value
            {"direction": None},
            {"line_cross": 1},                # int, not bool
            {"clip": "yes"},
            {"dwell_s": "five"},
            {"dwell_s": -1},
            {"pre_s": True},                  # bool is not a number here
            {"post_s": -0.5},
        ]
        for extra in bad_msgs:
            action, payload = le.validate_control({"type": "set_triggers", **extra})
            assert action == "unknown", extra
            assert payload == {}

    def test_set_watch_valid(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "type": "set_watch", "enabled": True, "condition": "any person visible",
            "interval_s": 10, "model": "lfm3b",
        })
        assert action == "set_watch"
        assert payload == {
            "enabled": True, "condition": "any person visible",
            "interval_s": 10.0, "model": "lfm3b",
        }
        # Interval bounds 1..30 inclusive on both ends.
        for iv in (1, 30, 4.5):
            assert le.validate_control({"type": "set_watch", "interval_s": iv})[0] == "set_watch"

    def test_set_watch_invalid(self):
        import visionbrain.live_engine as le

        bad_msgs = [
            {"model": "gemma"},               # watch models are lfm|lfm3b only
            {"model": None},
            {"interval_s": 0},                # below 1
            {"interval_s": 31},               # above 30
            {"interval_s": "fast"},
            {"interval_s": True},
            {"enabled": 1},                   # not a bool
            {"condition": 42},                # not a string
        ]
        for extra in bad_msgs:
            action, payload = le.validate_control({"type": "set_watch", **extra})
            assert action == "unknown", extra
            assert payload == {}


class TestRectZone:
    def test_contains(self):
        import visionbrain.live_engine as le

        zone = le.RectZone(10, 20, 30, 40, "yard")
        assert zone.contains(10, 20) and zone.contains(30, 40)  # edges inclusive
        assert zone.contains(20, 30)
        assert not zone.contains(9.9, 30) and not zone.contains(20, 40.1)

    def test_enter_exit_transitions(self):
        import visionbrain.live_engine as le

        zone = le.RectZone(0, 0, 10, 10, "yard")
        # First sighting inside → exactly one enter event.
        ev = zone.update(1, True, 1.0)
        assert ev == {"kind": "zone_enter", "zone": "yard", "track_id": 1,
                      "detail": "track 1 entered", "ts": 1.0}
        # Staying inside → silent.
        assert zone.update(1, True, 2.0) is None
        # Leaving → one exit event.
        ev = zone.update(1, False, 3.0)
        assert ev["kind"] == "zone_exit"
        assert ev["detail"] == "track 1 exited"
        # Staying outside → silent; re-entering fires again.
        assert zone.update(1, False, 4.0) is None
        assert zone.update(1, True, 5.0)["kind"] == "zone_enter"

        # Tracks are independent.
        assert zone.update(2, True, 6.0)["track_id"] == 2
        assert zone.update(1, True, 7.0) is None


class TestDwellTracker:
    def test_fires_once_then_re_arms(self):
        import visionbrain.live_engine as le

        tracker = le.DwellTracker(5.0)
        assert tracker.update(7, True, 10.0) is None          # timer starts
        assert tracker.update(7, True, 13.0) is None          # 3s < 5s
        ev = tracker.update(7, True, 15.0)                    # 5s reached
        assert ev == {"kind": "dwell", "track_id": 7,
                      "detail": "track 7 stationary >= 5s", "ts": 15.0}
        # Still stationary → fired once per stretch, no more events.
        assert tracker.update(7, True, 20.0) is None
        assert tracker.update(7, True, 100.0) is None
        # Movement clears the stretch; a new stretch can fire again.
        assert tracker.update(7, False, 101.0) is None
        assert tracker.update(7, True, 102.0) is None
        ev2 = tracker.update(7, True, 107.5)
        assert ev2 is not None and ev2["track_id"] == 7

    def test_tracks_independent_and_disabled_at_zero(self):
        import visionbrain.live_engine as le

        tracker = le.DwellTracker(2.0)
        assert tracker.update(1, True, 0.0) is None
        assert tracker.update(1, True, 2.0) is not None
        # A different track has its own timer.
        assert tracker.update(2, True, 0.5) is None
        assert tracker.update(2, True, 2.5) is not None
        # Movement by track 1 does not affect track 2's fired state.
        assert tracker.update(1, False, 3.0) is None
        assert tracker.update(2, True, 3.0) is None

        # dwell_s = 0 means disabled: timers start but never fire.
        off = le.DwellTracker(0.0)
        assert off.update(3, True, 0.0) is None
        assert off.update(3, True, 999.0) is None


class TestDirectionTriggerState:
    def test_none_never_fires(self):
        import visionbrain.live_engine as le

        state = le.DirectionTriggerState()
        for d in ("north", "east", "any-label", "unknown", "stationary"):
            assert state.update(1, d, "none", 1.0) is None

    def test_any_semantics(self):
        import visionbrain.live_engine as le

        state = le.DirectionTriggerState()
        # unknown / stationary never fire, even under "any".
        assert state.update(1, "unknown", "any", 1.0) is None
        assert state.update(1, "stationary", "any", 2.0) is None
        # Any real compass heading fires once...
        ev = state.update(1, "east", "any", 3.0)
        assert ev["kind"] == "direction"
        assert ev["track_id"] == 1 and ev["direction"] == "east"
        assert "east" in ev["detail"]
        # ...and the same heading does not re-fire...
        assert state.update(1, "east", "any", 4.0) is None
        assert state.update(1, "east", "any", 5.0) is None
        # ...but a heading change re-arms: a new direction fires again.
        ev2 = state.update(1, "north", "any", 6.0)
        assert ev2 is not None and ev2["direction"] == "north"
        # And returning to a previous heading fires again after the change.
        ev3 = state.update(1, "east", "any", 7.0)
        assert ev3 is not None and ev3["direction"] == "east"

    def test_fixed_direction_match(self):
        import visionbrain.live_engine as le

        state = le.DirectionTriggerState()
        # Non-matching headings are silent.
        assert state.update(2, "west", "north", 1.0) is None
        assert state.update(2, "stationary", "north", 2.0) is None
        # Match fires once...
        ev = state.update(2, "north", "north", 3.0)
        assert ev is not None and ev["direction"] == "north"
        assert state.update(2, "north", "north", 4.0) is None
        # ...and re-arms only via a direction change (west then north again).
        assert state.update(2, "west", "north", 5.0) is None
        ev2 = state.update(2, "north", "north", 6.0)
        assert ev2 is not None

    def test_tracks_independent(self):
        import visionbrain.live_engine as le

        state = le.DirectionTriggerState()
        assert state.update(1, "east", "any", 1.0) is not None
        # Another track heading the same way fires on its own.
        assert state.update(2, "east", "any", 2.0) is not None
        # Track 1 holding its heading stays silent while 2 re-arms via change.
        assert state.update(1, "east", "any", 3.0) is None
        assert state.update(2, "south", "any", 4.0) is not None


# ──────────────────────────────────────────────────────────────────────────────
# VB_TOKEN auth at the WebSocket door (CI-safe: fastapi only, no MLX/weights)
# ──────────────────────────────────────────────────────────────────────────────

class TestLiveEngineAuth:
    """live_ws honors VB_TOKEN via ``?token=`` (browsers cannot set WS headers).

    With VB_TOKEN set, a connect without (or with a wrong) token is denied
    at the handshake — live_ws closes pre-accept with code 4401, which the
    Starlette TestClient surfaces as ``WebSocketDisconnect`` carrying that
    code (verified against the installed starlette version). With VB_TOKEN
    unset, no token is needed and the standard "local engine ready" status
    arrives immediately.
    """

    @staticmethod
    def _client():
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import visionbrain.live_engine as le

        app = FastAPI()
        app.include_router(le.router)
        return TestClient(app)

    def test_denied_without_token(self, monkeypatch):
        from starlette.websockets import WebSocketDisconnect

        monkeypatch.setenv("VB_TOKEN", "sekret")
        client = self._client()
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/api/live/ws"):
                pass  # never accepted — handshake denied pre-accept
        assert excinfo.value.code == 4401

    def test_denied_with_wrong_token(self, monkeypatch):
        from starlette.websockets import WebSocketDisconnect

        monkeypatch.setenv("VB_TOKEN", "sekret")
        client = self._client()
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/api/live/ws?token=wrong"):
                pass
        assert excinfo.value.code == 4401

    def test_accepted_with_correct_token(self, monkeypatch):
        monkeypatch.setenv("VB_TOKEN", "sekret")
        client = self._client()
        with client.websocket_connect("/api/live/ws?token=sekret") as ws:
            msg = json.loads(ws.receive_text())
        assert msg == {"type": "status", "note": "local engine ready"}

    def test_no_token_needed_when_unset(self, monkeypatch):
        monkeypatch.delenv("VB_TOKEN", raising=False)
        client = self._client()
        with client.websocket_connect("/api/live/ws") as ws:
            msg = json.loads(ws.receive_text())
        assert msg == {"type": "status", "note": "local engine ready"}


class TestValidateControlTuning:
    """Live-tunable knobs (start keys + set_* controls) and the hub-protocol
    controls the local engine now answers instead of silently no-oping."""

    def test_start_tuning_keys_pass(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "action": "start", "source": "file", "file_id": "abc",
            "prompts": ["person"], "threshold": 0.3, "detect_every": 4,
            "resolution": 720, "backbone_every": 5,
            "jpeg_quality": 55, "send_width": 960, "task": "detect",
        })
        assert action == "start"
        assert payload["backbone_every"] == 5
        assert payload["jpeg_quality"] == 55
        assert payload["send_width"] == 960
        assert payload["task"] == "detect"

    def test_start_tuning_bogus_values_dropped(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "action": "start", "source": "webcam", "camera": 0,
            "prompts": ["person"], "jpeg_quality": 200, "send_width": 10,
            "task": "yolo", "backbone_every": 0, "threshold": 5,
        })
        assert action == "start"
        for key in ("jpeg_quality", "send_width", "task", "backbone_every", "threshold"):
            assert key not in payload

    def test_set_threshold(self):
        import visionbrain.live_engine as le

        assert le.validate_control({"type": "set_threshold", "threshold": 0.4}) == (
            "set_threshold", {"threshold": 0.4},
        )
        for bad in (0, 1, -0.1, "0.4", True, None):
            assert le.validate_control({"type": "set_threshold", "threshold": bad}) == (
                "unknown", {},
            )

    def test_set_stream(self):
        import visionbrain.live_engine as le

        assert le.validate_control(
            {"type": "set_stream", "jpeg_quality": 50, "send_width": 640}
        ) == ("set_stream", {"jpeg_quality": 50, "send_width": 640})
        assert le.validate_control({"type": "set_stream", "jpeg_quality": 80}) == (
            "set_stream", {"jpeg_quality": 80},
        )
        # empty payload, out-of-range quality, non-int quality, bad width
        assert le.validate_control({"type": "set_stream"}) == ("unknown", {})
        assert le.validate_control({"type": "set_stream", "jpeg_quality": 29}) == ("unknown", {})
        assert le.validate_control({"type": "set_stream", "jpeg_quality": 96}) == ("unknown", {})
        assert le.validate_control({"type": "set_stream", "jpeg_quality": 55.5}) == ("unknown", {})
        assert le.validate_control({"type": "set_stream", "send_width": 100}) == ("unknown", {})

    def test_set_task(self):
        import visionbrain.live_engine as le

        assert le.validate_control({"type": "set_task", "task": "detect"}) == (
            "set_task", {"task": "detect"},
        )
        assert le.validate_control({"type": "set_task", "task": "segment"}) == (
            "set_task", {"task": "segment"},
        )
        assert le.validate_control({"type": "set_task", "task": "yolo"}) == ("unknown", {})

    def test_set_engine_accepts_bools_and_lfm_model(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({
            "type": "set_engine", "sam": True, "falcon": False, "lfm_model": "lfm3b",
        })
        assert action == "set_engine"
        assert payload == {"sam": True, "falcon": False, "lfm_model": "lfm3b"}
        assert le.validate_control({"type": "set_engine", "sam": "yes"}) == ("unknown", {})
        assert le.validate_control({"type": "set_engine", "lfm_model": "gemma"}) == ("unknown", {})

    def test_set_vlm(self):
        import visionbrain.live_engine as le

        assert le.validate_control({"type": "set_vlm", "model": "lfm"}) == (
            "set_vlm", {"model": "lfm"},
        )
        assert le.validate_control({"type": "set_vlm", "model": "  "}) == ("unknown", {})
        assert le.validate_control({"type": "set_vlm", "model": 3}) == ("unknown", {})

    def test_ask(self):
        import visionbrain.live_engine as le

        action, payload = le.validate_control({"type": "ask", "question": "  what do you see? "})
        assert (action, payload) == ("ask", {"question": "what do you see?"})
        assert le.validate_control({"type": "ask", "question": "   "}) == ("unknown", {})
        assert le.validate_control({"type": "ask", "question": "x" * 501}) == ("unknown", {})
        assert le.validate_control({"type": "ask"}) == ("unknown", {})

    def test_report(self):
        import visionbrain.live_engine as le

        assert le.validate_control(
            {"type": "report", "summary": "2x person", "report_type": "field"}
        ) == ("report", {"summary": "2x person", "report_type": "field"})
        assert le.validate_control({"type": "report"}) == (
            "report", {"summary": "", "report_type": "field"},
        )
        assert le.validate_control({"type": "report", "summary": 42}) == ("unknown", {})
        assert le.validate_control({"type": "report", "report_type": ""}) == ("unknown", {})


class TestWorkerTunables:
    """Live-tunable worker state — set_* mutators clamp and take effect."""

    @staticmethod
    def _worker():
        import asyncio
        import threading

        import visionbrain.live_engine as le

        # A RUNNING loop: push() delivers via call_soon_threadsafe, which
        # needs the loop alive for the sentinel to reach the queue.
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        sink = le._ClientSink(loop)
        worker = le._EngineWorker(
            {"source": "webcam", "camera": 0, "prompts": ["person"]},
            {sink},
        )
        return worker, sink

    def test_stream_tuning_clamped(self):
        worker, _sink = self._worker()
        worker.set_stream(jpeg_quality=1000, send_width=1)
        assert worker._jpeg_quality == 95
        assert worker._send_width == 256
        worker.set_stream(jpeg_quality=50, send_width=1920)
        assert worker._jpeg_quality == 50
        assert worker._send_width == 1920

    def test_threshold_and_task(self):
        worker, _sink = self._worker()
        worker.set_threshold(0.6)
        worker.set_task("detect")
        assert worker._threshold == 0.6
        assert worker._task == "detect"

    def test_push_frame_coalesces_to_one_slot(self):
        import time

        import visionbrain.live_engine as le

        worker, sink = self._worker()
        worker.push_frame(b"f1")
        worker.push_frame(b"f2")
        worker.push_frame(b"f3")
        time.sleep(0.1)  # let the loop thread run put_nowait
        # The sink keeps just the newest frame; the queue holds exactly one
        # sentinel for the whole burst — a stalled sender can never grow a
        # stale-JPEG backlog.
        assert sink.pending_frame == b"f3"
        assert sink.queue.qsize() == 1
        assert sink.queue.get_nowait() is le._FRAME_SENTINEL
        # Sender drains the slot; the next push re-arms the sentinel.
        sink.pending_frame = None
        worker.push_frame(b"f4")
        time.sleep(0.1)
        assert sink.pending_frame == b"f4"
        assert sink.queue.qsize() == 1

    def test_push_fans_out_to_all_viewers(self):
        import asyncio
        import threading
        import time

        import visionbrain.live_engine as le

        # Two attached viewers both receive JSON broadcasts and frames; the
        # latest-frame-wins coalescing stays per viewer.
        loops, sinks = [], []
        for _ in range(2):
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True).start()
            loops.append(loop)
            sinks.append(le._ClientSink(loop))
        worker = le._EngineWorker(
            {"source": "webcam", "camera": 0, "prompts": ["person"]},
            set(sinks),
        )
        worker.push({"type": "status", "note": "hi"})
        worker.push_frame(b"f1")
        worker.push_frame(b"f2")
        time.sleep(0.1)
        for sink in sinks:
            assert sink.queue.get_nowait() == {"type": "status", "note": "hi"}
            assert sink.queue.get_nowait() is le._FRAME_SENTINEL
            assert sink.pending_frame == b"f2"
            assert sink.queue.qsize() == 0

    def test_last_detection_records_shape(self):
        worker, _sink = self._worker()
        with worker._frame_lock:
            worker._last_frame_items = [{
                "label": "person", "score": 0.9,
                "box": [0.1, 0.2, 0.3, 0.4], "track_id": 1,
            }]
        records = worker.last_detection_records()
        assert records == [{
            "label": "person", "score": 0.9,
            "centroid_norm": {"x": pytest.approx(0.2), "y": pytest.approx(0.3)},
            "source": "sam",
        }]


# ──────────────────────────────────────────────────────────────────────────────
# SIMPLIFICATION_SPEC regressions (sections 3-6, server side) — shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_loop_sink(le):
    """A worker sink wired to a RUNNING event loop on its own daemon thread.

    Returns ``(loop, sink, stop)`` — the caller MUST invoke ``stop()`` so no
    loop thread outlives the test.
    """
    import asyncio

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    def stop() -> None:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2.0)

    return loop, le._ClientSink(loop), stop


def _writer_thread_count() -> int:
    """Number of live ``live-capture-writer`` threads right now."""
    return len([t for t in threading.enumerate() if t.name == "live-capture-writer"])


def _recv_until(ws, predicate, limit: int = 8) -> dict:
    """Read JSON control responses until ``predicate`` matches (bounded)."""
    for _ in range(limit):
        msg = json.loads(ws.receive_text())
        if predicate(msg):
            return msg
    pytest.fail("expected websocket message not received")


def _wait_worker(le, timeout: float = 5.0):
    """Block until the WS handler has installed a live worker; return it."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        worker = le._worker
        if worker is not None and worker.thread is not None and worker.thread.is_alive():
            return worker
        time.sleep(0.01)
    pytest.fail("engine worker did not start")


def _fake_run_patch(monkeypatch) -> None:
    """Replace ``_EngineWorker.run`` so a WS start needs no cv2/MLX.

    The fake idles until ``stop_event`` (keeping the worker ``_alive``) and
    mimics the real exit contract: ``engine_stopped`` then writer shutdown.
    """
    import visionbrain.live_engine as le

    def fake_run(self) -> None:
        self.stop_event.wait(5.0)
        self.push({"type": "engine_stopped"})
        self._shutdown_capture_writer()

    monkeypatch.setattr(le._EngineWorker, "run", fake_run)


@pytest.fixture
def ws_clean(monkeypatch):
    """Isolate the live-engine module globals for one WS-level test."""
    import visionbrain.live_engine as le

    monkeypatch.setattr(le, "_worker", None)
    monkeypatch.setattr(le, "_sinks", set())
    monkeypatch.setattr(le, "_pending_zones", None)
    monkeypatch.setattr(le, "_pending_triggers", None)
    monkeypatch.setattr(le, "_pending_targets", None)
    monkeypatch.setattr(le, "_ask_busy", False)
    return le


def _patch_vlm(monkeypatch, *, ask_result="ok", ask_error=None,
               report_result="report ok", report_error=None) -> dict:
    """Mock ``vlm_registry`` ask/report; return the recorded call args."""
    import visionbrain.vlm_registry as vlmr

    calls: dict = {"ask": [], "generate_report": []}

    def fake_ask(question, detections=None, prompts=None, image=None):
        calls["ask"].append({
            "question": question, "detections": detections,
            "prompts": prompts, "image": image,
        })
        if ask_error is not None:
            raise ask_error
        return ask_result

    def fake_generate_report(summary, report_type="field", image=None):
        calls["generate_report"].append({
            "summary": summary, "report_type": report_type, "image": image,
        })
        if report_error is not None:
            raise report_error
        return report_result

    monkeypatch.setattr(vlmr, "ask", fake_ask)
    monkeypatch.setattr(vlmr, "generate_report", fake_generate_report)
    monkeypatch.setattr(vlmr, "current_key", lambda: "gemma")
    return calls


def _set_evidence(worker, items=None, frame_id: int = 42):
    """Simulate one completed detect pass; returns the frame sentinel."""
    frame = object()
    with worker._frame_lock:
        worker._latest_frame = (frame_id, frame)
        worker._last_frame_items = list(items if items is not None else [])
    return frame


def _ws_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import visionbrain.live_engine as le

    app = FastAPI()
    app.include_router(le.router)
    return TestClient(app)


class TestBackboneDeprecation:
    """Spec section 3: features are recomputed every detection pass; the
    legacy ``backbone_every`` start key is accepted, stripped from the
    worker payload, and earns a deprecation status note."""

    def test_worker_stores_no_backbone_state(self):
        import visionbrain.live_engine as le

        worker = le._EngineWorker(
            {"source": "webcam", "camera": 0, "prompts": ["person"],
             "backbone_every": 2},
            set(),
        )
        # The legacy knob is inert: validated upstream, never stored.
        assert not hasattr(worker, "_backbone_every")
        assert worker.cfg["backbone_every"] == 2  # carried but unread

    def test_start_with_backbone_every_sends_deprecation_note(self, ws_clean, monkeypatch):
        le = ws_clean
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text()) == {
                "type": "status", "note": "local engine ready",
            }
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["person"], "backbone_every": 2,
            }))
            note = _recv_until(
                ws, lambda m: "backbone_every" in str(m.get("note", "")),
            )
            assert note["type"] == "status"
            assert "deprecated" in note["note"]
            assert "every detection pass" in note["note"]
            worker = _wait_worker(le)
            # The worker payload never sees the key.
            assert "backbone_every" not in worker.cfg
            worker.stop_event.set()

    def test_start_without_backbone_every_sends_no_note(self, ws_clean, monkeypatch):
        le = ws_clean
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["person"],
            }))
            worker = _wait_worker(le)
            ws.send_text(json.dumps({"type": "stop"}))
            msgs = [_recv_until(ws, lambda m: m.get("type") == "engine_stopped")]
            notes = [m.get("note", "") for m in msgs if m.get("type") == "status"]
            assert all("deprecated" not in n for n in notes)
            assert "backbone_every" not in worker.cfg


class TestEvidenceSnapshot:
    """Spec section 6 (server): one consistent evidence snapshot, pause
    clears it, and report counts are derived server-side."""

    @staticmethod
    def _worker():
        import visionbrain.live_engine as le

        return le._EngineWorker(
            {"source": "webcam", "camera": 0, "prompts": ["car"]}, set()
        )

    @staticmethod
    def _items() -> list[dict]:
        return [
            {"label": "car", "score": 0.9, "box": [0.0, 0.0, 0.2, 0.2], "track_id": 1},
            {"label": "car", "score": 0.8, "box": [0.2, 0.2, 0.4, 0.4], "track_id": 2},
            {"label": "car", "score": 0.7, "box": [0.4, 0.4, 0.6, 0.6], "track_id": 3},
            {"label": "person", "score": 0.6, "box": [0.6, 0.6, 0.8, 0.8], "track_id": 4},
        ]

    def test_snapshot_none_without_observation(self):
        worker = self._worker()
        assert worker.evidence_snapshot() is None

    def test_snapshot_one_consistent_triple(self):
        worker = self._worker()
        frame = object()
        with worker._frame_lock:
            worker._latest_frame = (7, frame)
            worker._last_frame_items = self._items()
        snap = worker.evidence_snapshot()
        assert snap is not None
        assert snap["frame_id"] == 7
        assert snap["frame"] is frame
        assert snap["prompts"] == ["car"]
        assert snap["records"][0] == {
            "label": "car", "score": 0.9,
            "centroid_norm": {"x": pytest.approx(0.1), "y": pytest.approx(0.1)},
            "source": "sam",
        }
        assert len(snap["records"]) == 4

    def test_observed_empty_set_is_valid_evidence(self):
        worker = self._worker()
        # The engine ran a real pass and saw NOTHING — valid observation,
        # distinct from having no observation at all.
        with worker._frame_lock:
            worker._latest_frame = (3, object())
            worker._last_frame_items = []
        snap = worker.evidence_snapshot()
        assert snap is not None
        assert snap["records"] == []

    def test_pause_clears_evidence_and_does_not_resurrect(self):
        worker = self._worker()
        with worker._frame_lock:
            worker._latest_frame = (1, object())
            worker._last_frame_items = self._items()
        worker.set_prompts([])  # pause — old counts must not survive
        assert worker.get_latest_frame() is None
        assert worker.evidence_snapshot() is None
        worker.set_prompts(["car"])  # resume — still no NEW observation
        assert worker.evidence_snapshot() is None
        # A non-empty set_prompts on live evidence keeps it.
        with worker._frame_lock:
            worker._latest_frame = (2, object())
        worker.set_prompts(["truck"])
        assert worker.evidence_snapshot() is not None

    def test_counts_summary_server_side(self):
        import visionbrain.live_engine as le

        assert le._counts_summary([]) == "no objects observed"
        records = le._records_from_items(self._items())
        assert le._counts_summary(records) == "3x car, 1x person"


class TestClipWriter:
    """Spec section 4: lazy writer start, ONE capture slot across
    collect + queue + encode, sentinel shutdown on every exit path."""

    @staticmethod
    def _writer_threads() -> list:
        return [t for t in threading.enumerate() if t.name == "live-capture-writer"]

    @staticmethod
    def _backdate_due(worker) -> None:
        """Make the pending capture's post-roll deadline already passed."""
        with worker._state_lock:
            worker._capture["deadline"] = 0.0

    @staticmethod
    def _await_slot_release(worker, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while worker._capture is not None and time.monotonic() < deadline:
            time.sleep(0.005)
        assert worker._capture is None

    @pytest.fixture
    def clip_env(self, tmp_path, monkeypatch):
        import visionbrain.live_engine as le

        monkeypatch.setattr(le, "_clips_dir", tmp_path / "clips")
        _loop, sink, stop = _make_loop_sink(le)
        worker = le._EngineWorker(
            {"source": "webcam", "camera": 0, "prompts": ["person"]}, {sink}
        )
        yield worker, sink
        # Cleanup: sentinel-stop any writer the test started, then the loop.
        worker._shutdown_capture_writer()
        stop()

    def test_constructor_starts_no_writer_thread(self, clip_env):
        worker, _sink = clip_env
        assert worker._capture_thread is None
        assert self._writer_threads() == []

    def test_writer_starts_lazily_and_slot_spans_encode(self, clip_env):
        worker, _sink = clip_env
        baseline = len(self._writer_threads())
        seen_threads: list[str] = []
        encode_started = threading.Event()
        release_encode = threading.Event()

        def controlled_encode(capture: dict) -> None:
            seen_threads.append(threading.current_thread().name)
            encode_started.set()
            release_encode.wait(5.0)

        worker._write_capture_file = controlled_encode

        assert worker.request_capture("line_cross", 100.0) is True
        assert worker._capture_thread is None  # still lazy while collecting
        self._backdate_due(worker)
        worker._dispatch_due_capture()
        assert encode_started.wait(5.0)  # writer picked it up
        # ...the slot stays claimed across queued work AND encoding...
        assert worker._capture is not None
        assert worker._capture["state"] == "encoding"
        assert worker.request_capture("dwell", 101.0) is False
        # ...and the writer releases it after a successful encode.
        release_encode.set()
        self._await_slot_release(worker)
        worker._shutdown_capture_writer()
        assert not worker._capture_thread.is_alive()
        assert len(self._writer_threads()) == baseline
        # Encoding never executed on the frame-processing thread.
        assert seen_threads == ["live-capture-writer"]

    def test_repeated_cycles_return_writer_count_to_baseline(self, clip_env):
        worker, _sink = clip_env
        baseline = len(self._writer_threads())
        worker._write_capture_file = lambda capture: None
        for _ in range(2):
            def fake__run() -> None:
                assert worker.request_capture("dwell", 100.0) is True
                self._backdate_due(worker)
                worker._dispatch_due_capture()

            worker._run = fake__run
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            thread.join(5.0)
            assert not thread.is_alive()
            # run()'s finally: slot released after encode + sentinel shutdown.
            assert worker._capture is None
            assert not worker._capture_thread.is_alive()
            assert len(self._writer_threads()) == baseline

    def test_burst_triggers_bounded_to_one_capture(self, clip_env):
        worker, _sink = clip_env
        release = threading.Event()
        encoded: list[str] = []

        def blocked_encode(capture: dict) -> None:
            encoded.append(capture["kind"])
            release.wait(5.0)

        worker._write_capture_file = blocked_encode

        assert worker.request_capture("line_cross", 100.0) is True
        # Burst while COLLECTING: later triggers lose the slot.
        assert worker.request_capture("direction", 100.5) is False
        assert worker.request_capture("dwell", 101.0) is False
        self._backdate_due(worker)
        worker._dispatch_due_capture()
        # Burst while QUEUED/ENCODING: still occupied.
        assert worker.request_capture("watch", 102.0) is False
        # Handoff is idempotent — no duplicate queue entry, single encode.
        worker._dispatch_due_capture()
        deadline = time.monotonic() + 5.0
        while not encoded and time.monotonic() < deadline:
            time.sleep(0.005)
        release.set()
        worker._shutdown_capture_writer()
        assert encoded == ["line_cross"]
        assert worker._capture is None

    def test_encode_failure_releases_slot(self, clip_env):
        worker, sink = clip_env

        def exploding_encode(capture: dict) -> None:
            raise RuntimeError("codec gone")

        worker._write_capture_file = exploding_encode
        assert worker.request_capture("dwell", 100.0) is True
        self._backdate_due(worker)
        worker._dispatch_due_capture()
        self._await_slot_release(worker)
        # The slot is genuinely reusable after the failure.
        assert worker.request_capture("dwell", 200.0) is True
        worker._shutdown_capture_writer()
        # And the failure surfaced as a status note, not a dead writer.
        time.sleep(0.1)
        notes = []
        while not sink.queue.empty():
            msg = sink.queue.get_nowait()
            if isinstance(msg, dict):
                notes.append(msg.get("note", ""))
        assert any("clip capture failed" in n and "codec gone" in n for n in notes)

    def test_shutdown_discards_incomplete_capture(self, clip_env):
        worker, _sink = clip_env
        encoded: list[dict] = []
        worker._write_capture_file = encoded.append
        worker._ensure_capture_writer()
        # post_s in the future → still collecting when the worker exits.
        assert worker.request_capture("dwell", time.time() + 60.0) is True
        worker._shutdown_capture_writer()
        assert worker._capture is None  # incomplete post-roll discarded
        assert encoded == []
        assert not worker._capture_thread.is_alive()
        assert worker.request_capture("dwell", time.time()) is True  # freed

    def test_shutdown_lets_completed_capture_finish_first(self, clip_env):
        worker, _sink = clip_env
        release = threading.Event()
        encoded: list[str] = []

        def blocked_encode(capture: dict) -> None:
            encoded.append(capture["kind"])
            release.wait(5.0)

        worker._write_capture_file = blocked_encode
        assert worker.request_capture("line_cross", 100.0) is True
        self._backdate_due(worker)
        worker._dispatch_due_capture()  # completed capture accepted
        done = threading.Event()

        def do_shutdown() -> None:
            worker._shutdown_capture_writer()  # sentinel AFTER the capture
            done.set()

        stopper = threading.Thread(target=do_shutdown, daemon=True)
        stopper.start()
        time.sleep(0.05)  # let the shutdown queue its sentinel
        release.set()     # the accepted capture finishes encoding
        assert done.wait(10.0)
        stopper.join(2.0)
        assert encoded == ["line_cross"]
        assert worker._capture is None  # slot released afterwards
        assert not worker._capture_thread.is_alive()

    def test_worker_exit_discards_incomplete_capture(self, clip_env):
        worker, _sink = clip_env

        def must_not_encode(capture: dict) -> None:
            pytest.fail("incomplete capture reached the encoder")

        worker._write_capture_file = must_not_encode
        worker._ensure_capture_writer()

        def fake__run() -> None:
            # Capture armed, post-roll never completes, loop ends.
            assert worker.request_capture("dwell", time.time() + 60.0) is True

        worker._run = fake__run
        thread = threading.Thread(target=worker.run, daemon=True)
        thread.start()
        thread.join(5.0)
        assert not thread.is_alive()
        assert worker._capture is None
        assert not worker._capture_thread.is_alive()


class TestDetectionsCoalescing:
    """Spec section 5: ``detections`` join frames in the per-viewer
    latest-wins slot (empty sets included); in-order messages are never
    dropped and one slow viewer never delays another."""

    @pytest.fixture
    def co_env(self):
        import visionbrain.live_engine as le

        _loop, sink, stop = _make_loop_sink(le)
        worker = le._EngineWorker(
            {"source": "webcam", "camera": 0, "prompts": ["person"]}, {sink}
        )
        yield worker, sink
        stop()

    def test_burst_keeps_only_latest(self, co_env):
        import visionbrain.live_engine as le

        worker, sink = co_env
        worker.push_detections([{"label": "a"}])
        worker.push_detections([{"label": "b"}])
        worker.push_detections([{"label": "c"}])
        time.sleep(0.05)
        assert sink.pending_detections == [{"label": "c"}]
        assert sink.queue.qsize() == 1
        assert sink.queue.get_nowait() is le._DETECTIONS_SENTINEL

    def test_empty_supersedes_nonempty(self, co_env):
        worker, sink = co_env
        worker.push_detections([{"label": "car"}])
        worker.push_detections([])  # "drop stale boxes" is a real value
        time.sleep(0.05)
        assert sink.pending_detections == []
        assert sink.queue.qsize() == 1  # one sentinel for the whole burst

    def test_nonempty_after_empty_replaces(self, co_env):
        worker, sink = co_env
        worker.push_detections([])
        worker.push_detections([{"label": "car", "score": 0.9}])
        time.sleep(0.05)
        assert sink.pending_detections == [{"label": "car", "score": 0.9}]

    def test_events_and_status_preserved_in_order(self, co_env):
        import visionbrain.live_engine as le

        worker, sink = co_env
        worker.push({"type": "status", "note": "s1"})
        worker.push_detections([{"label": "a"}])
        worker.push({"type": "event", "event": {"kind": "dwell"}})
        worker.push_detections([])  # supersedes "a", no new sentinel
        worker.push({"type": "capture", "clip": {"name": "x.mp4"}})
        time.sleep(0.05)
        drained = []
        while not sink.queue.empty():
            drained.append(sink.queue.get_nowait())
        # Every in-order message survived; exactly one detections sentinel.
        assert drained == [
            {"type": "status", "note": "s1"},
            le._DETECTIONS_SENTINEL,
            {"type": "event", "event": {"kind": "dwell"}},
            {"type": "capture", "clip": {"name": "x.mp4"}},
        ]
        assert sink.pending_detections == []

    def test_slow_viewer_does_not_delay_another(self):
        import visionbrain.live_engine as le

        loops, sinks, stops = [], [], []
        try:
            for _ in range(2):
                loop, sink, stop = _make_loop_sink(le)
                loops.append(loop)
                sinks.append(sink)
                stops.append(stop)
            worker = le._EngineWorker(
                {"source": "webcam", "camera": 0, "prompts": ["person"]},
                set(sinks),
            )
            # Neither viewer drains — fan-out must not block or backlog.
            worker.push({"type": "status", "note": "hi"})
            worker.push_detections([{"label": "x"}])
            worker.push_frame(b"f1")
            time.sleep(0.05)
            for sink in sinks:
                assert sink.pending_detections == [{"label": "x"}]
                assert sink.pending_frame == b"f1"
                # One sentinel per replaceable stream — bounded per viewer.
                assert sink.queue.qsize() == 3
        finally:
            for stop in stops:
                stop()

    def test_sender_ships_latest_wire_messages(self):
        import asyncio

        import visionbrain.live_engine as le

        class FakeWS:
            def __init__(self) -> None:
                self.sent: list[tuple[str, object]] = []

            async def send_bytes(self, data) -> None:
                self.sent.append(("bin", bytes(data)))

            async def send_text(self, text: str) -> None:
                self.sent.append(("txt", json.loads(text)))

        async def main() -> list:
            loop = asyncio.get_running_loop()
            sink = le._ClientSink(loop)
            sink.push({"type": "event", "event": {"kind": "dwell"}})
            sink.push_detections([{"label": "car"}])
            sink.push_detections([])  # empty supersedes the car
            sink.push_frame(b"f1")
            sink.push_frame(b"f2")
            ws = FakeWS()
            task = asyncio.create_task(le._sender(ws, sink))
            await asyncio.sleep(0.05)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return ws.sent

        sent = asyncio.run(main())
        assert sent == [
            ("txt", {"type": "event", "event": {"kind": "dwell"}}),
            ("txt", {"type": "detections", "items": []}),  # empty set shipped
            ("bin", b"f2"),  # latest frame wins
        ]


class TestAskReportWire:
    """Spec section 6 (server side): pinned ``error`` shape on the ask/report
    path, evidence snapshot + grounding rule, server-derived report counts.
    Uses a fake engine run and mocked ``vlm_registry`` — no MLX anywhere."""

    def test_ask_report_no_engine_error_shape(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text()) == {
                "type": "status", "note": "local engine ready",
            }
            ws.send_text(json.dumps({"type": "ask", "question": "what?"}))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert "no engine running" in msg["error"]
        # Same pinned shape for report.
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({"type": "report", "summary": "x"}))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert "no engine running" in msg["error"]
        assert calls["ask"] == []
        assert calls["generate_report"] == []

    def test_ask_rejects_without_observation_or_inference(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch)
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car"],
            }))
            _wait_worker(le)
            # No detect pass has run: grounded rejection BEFORE the slot.
            ws.send_text(json.dumps({"type": "ask", "question": "what?"}))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert "no current observation" in msg["error"]
            assert calls["ask"] == []
            # The shared inference slot was never claimed.
            assert le._claim_ask_slot() is True
            le._release_ask_slot()

    def test_pause_clears_evidence_and_ask_refuses(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch)
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car"],
            }))
            worker = _wait_worker(le)
            _set_evidence(worker, items=[{"label": "car", "score": 0.9,
                                          "box": [0, 0, 0.2, 0.2]}])
            assert worker.evidence_snapshot() is not None
            ws.send_text(json.dumps({"type": "set_prompts", "prompts": []}))
            note = json.loads(ws.receive_text())
            assert note == {"type": "status", "note": "detection paused (no prompts)"}
            # Pause cleared the stored evidence entirely.
            assert worker.get_latest_frame() is None
            assert worker.evidence_snapshot() is None
            ws.send_text(json.dumps({"type": "ask", "question": "how many?"}))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert "no current observation" in msg["error"]
            assert calls["ask"] == []
            # Re-arming prompts does not resurrect the cleared evidence.
            ws.send_text(json.dumps({"type": "set_prompts", "prompts": ["car"]}))
            note = json.loads(ws.receive_text())
            assert note == {"type": "status", "note": "prompts updated"}
            assert worker.evidence_snapshot() is None

    def test_busy_slot_error_shape(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch)
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car"],
            }))
            worker = _wait_worker(le)
            _set_evidence(worker)
            assert le._claim_ask_slot() is True  # someone else is inferring
            try:
                ws.send_text(json.dumps({"type": "ask", "question": "hello?"}))
                msg = json.loads(ws.receive_text())
                assert msg == {"type": "error", "error": "ask/report already running"}
                ws.send_text(json.dumps({"type": "report"}))
                msg = json.loads(ws.receive_text())
                assert msg == {"type": "error", "error": "ask/report already running"}
            finally:
                le._release_ask_slot()
            assert calls["ask"] == []
            assert calls["generate_report"] == []

    def test_ask_success_uses_dispatch_snapshot(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch, ask_result="a car")
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car"],
            }))
            worker = _wait_worker(le)
            frame = _set_evidence(
                worker,
                items=[{"label": "car", "score": 0.9, "box": [0, 0, 0.2, 0.2]}],
                frame_id=42,
            )
            ws.send_text(json.dumps(
                {"type": "ask", "question": "  what do you see? "}
            ))
            ack = json.loads(ws.receive_text())
            assert ack == {"type": "ask_ack", "model": "gemma"}
            answer = json.loads(ws.receive_text())
            assert answer == {"type": "answer", "answer": "a car"}
            # The background thread answered from the SNAPSHOT, not live state.
            assert calls["ask"][0]["question"] == "what do you see?"
            assert calls["ask"][0]["image"] is frame
            assert calls["ask"][0]["prompts"] == ["car"]
            assert calls["ask"][0]["detections"] == worker.evidence_snapshot()["records"]

    def test_ask_failure_pushes_pinned_error_and_releases_slot(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch, ask_error=RuntimeError("boom"))
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car"],
            }))
            worker = _wait_worker(le)
            _set_evidence(worker)
            ws.send_text(json.dumps({"type": "ask", "question": "hello?"}))
            assert json.loads(ws.receive_text())["type"] == "ask_ack"
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert msg["error"] == "ask failed: boom"
            # The failure released the shared slot.
            assert le._claim_ask_slot() is True
            le._release_ask_slot()
        assert calls["ask"]

    def test_report_counts_derived_server_side(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch, report_result="all quiet")
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car", "person"],
            }))
            worker = _wait_worker(le)
            items = (
                [{"label": "car", "score": 0.9, "box": [0, 0, 0.1, 0.1]}] * 3
                + [{"label": "person", "score": 0.8, "box": [0.2, 0.2, 0.3, 0.3]}]
            )
            frame = _set_evidence(worker, items=items)
            # Forged browser summary must not reach the model.
            ws.send_text(json.dumps({
                "type": "report", "summary": "999x bird", "report_type": "field",
            }))
            msg = json.loads(ws.receive_text())
            assert msg == {"type": "report_result", "text": "all quiet"}
            sent = calls["generate_report"][0]
            assert sent["summary"] == "3x car, 1x person"
            assert "bird" not in sent["summary"]
            assert sent["report_type"] == "field"
            assert sent["image"] is frame

    def test_report_empty_observation_says_so(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch, report_result="quiet scene")
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car"],
            }))
            worker = _wait_worker(le)
            # Engine ran a real pass and saw nothing — a valid observation.
            _set_evidence(worker, items=[])
            ws.send_text(json.dumps({"type": "report"}))
            msg = json.loads(ws.receive_text())
            assert msg == {"type": "report_result", "text": "quiet scene"}
            assert calls["generate_report"][0]["summary"] == "no objects observed"

    def test_report_failure_pushes_pinned_error(self, ws_clean, monkeypatch):
        le = ws_clean
        calls = _patch_vlm(monkeypatch, report_error=RuntimeError("vlm down"))
        _fake_run_patch(monkeypatch)
        client = _ws_client()
        with client.websocket_connect("/api/live/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "status"
            ws.send_text(json.dumps({
                "type": "start", "source": "webcam", "camera": 0,
                "prompts": ["car"],
            }))
            worker = _wait_worker(le)
            _set_evidence(worker)
            ws.send_text(json.dumps({"type": "report"}))
            msg = json.loads(ws.receive_text())
            assert msg == {"type": "error", "error": "report failed: vlm down"}
            # Slot released after the failure.
            assert le._claim_ask_slot() is True
            le._release_ask_slot()
