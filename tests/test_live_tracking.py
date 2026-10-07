"""LiveSamTracker cadence, freshness attribution and invalidation.

Uses injected backbone/detect callables (simulated perception) so the logic
runs without weights. The real-model path is covered by the bridge benchmark.
"""

import types

import numpy as np
import pytest

from visionbrain import live_tracking as lt


class Fakes:
    """Counts backbone and detect calls and returns a fixed box."""

    def __init__(self):
        self.backbones = []
        self.detects = 0

    def preprocess(self, processor, image):
        return image

    def backbone(self, model, pixels):
        self.backbones.append(pixels)
        return pixels

    def detect(self, predictor, features, prompts, size, threshold, cache):
        self.detects += 1
        return types.SimpleNamespace(
            scores=np.array([0.9]),
            boxes=np.array([[10.0, 10.0, 50.0, 50.0]]),
            labels=["car"],
            masks=None,
            track_ids=None,
        )


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    monkeypatch.setattr(lt, "_ensure_loaded", lambda *a: (None, None, None))


def make(fakes, **kwargs):
    kwargs.setdefault("detect_every", 2)
    kwargs.setdefault("backbone_every", 3)
    return lt.LiveSamTracker(
        backbone_fn=fakes.backbone,
        detect_fn=fakes.detect,
        preprocess_fn=fakes.preprocess,
        tracker=types.SimpleNamespace(update=lambda r: r),
        **kwargs,
    )


def run(tracker, frames, *, ms_per_frame=100, **kwargs):
    out = []
    for n in frames:
        out.append(
            tracker.step(f"img{n}", ["car"], "detect", 100, 100, n, n * ms_per_frame, **kwargs)
        )
    return out


def test_backbone_refreshes_every_backbone_every_detect_passes():
    fakes = Fakes()
    run(make(fakes, detect_every=4, backbone_every=6, max_backbone_age_ms=10**9), range(49))
    # detect passes at frames 0,4,8,...; backbone at passes 0,6,12 -> frames 0,24,48
    assert fakes.backbones == ["img0", "img24", "img48"]
    assert fakes.detects == 13


def test_cached_backbone_items_carry_the_backbone_frame_and_age():
    fakes = Fakes()
    out = run(make(fakes), range(5))
    fresh, held, cached = out[0][0], out[1][0], out[2][0]
    assert (fresh["observed_frame_id"], fresh["stale_ms"], fresh["track_state"]) == (0, 0, "active")
    assert held["observed_frame_id"] == 0 and held["stale_ms"] == 100
    # frame 2 reran DETR on the frame-0 backbone
    assert cached["observed_frame_id"] == 0
    assert cached["observed_timestamp_ms"] == 0
    assert cached["stale_ms"] == 200
    assert cached["track_state"] == "predicted"


def test_old_backbone_is_refreshed_before_the_cadence_says_so():
    fakes = Fakes()
    tracker = make(fakes, backbone_every=100, max_backbone_age_ms=250)
    out = run(tracker, range(6))
    assert fakes.backbones == ["img0", "img4"]
    assert out[4][0]["stale_ms"] == 0
    assert max(it["stale_ms"] for step in out for it in step) <= 250 + 100


def test_stale_age_survives_u32_timestamp_wrap():
    fakes = Fakes()
    tracker = make(fakes)
    near_wrap = (1 << 32) - 100
    tracker.step("a", ["car"], "detect", 100, 100, 1, near_wrap)
    held = tracker.step("b", ["car"], "detect", 100, 100, 2, 100)
    assert held[0]["stale_ms"] == 200


@pytest.mark.parametrize(
    "change",
    [
        {"prompts": ["truck"]},
        {"task": "segment"},
        {"width": 200},
        {"source_key": ("scout", 2)},
    ],
)
def test_incompatible_config_drops_cache_and_held_items(change):
    fakes = Fakes()
    tracker = make(fakes, detect_every=5)
    args = dict(prompts=["car"], task="detect", width=100, height=100, source_key=("scout", 1))
    tracker.step("a", args["prompts"], args["task"], args["width"], args["height"], 1, 0, source_key=args["source_key"])
    args.update(change)
    out = tracker.step("b", args["prompts"], args["task"], args["width"], args["height"], 2, 100, source_key=args["source_key"])
    assert fakes.backbones == ["a", "b"]
    assert out and out[0]["observed_frame_id"] == 2 and out[0]["stale_ms"] == 0


def test_same_config_keeps_cache():
    fakes = Fakes()
    tracker = make(fakes, detect_every=5)
    for n in (1, 2):
        tracker.step("x", ["car"], "detect", 100, 100, n, n * 100, source_key=("scout", 1))
    assert len(fakes.backbones) == 1
