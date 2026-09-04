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
            {"source": "webcam", "camera": 0, "prompts": ["person"]}, None, None
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
