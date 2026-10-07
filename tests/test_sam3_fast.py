"""Parity of detect_multi(fast=True) against the stock mlx_vlm path.

Runs the real SAM 3.1 checkpoint on a real photo. Skipped when the weights
or the MLX runtime are unavailable. No simulated perception.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visionbrain.loader import _check_mlx
from visionbrain.sam3_inference import detect_multi, sam31_available

IMAGE = Path(__file__).resolve().parents[1] / "assets" / "samples" / "friends_people_som.jpg"

pytestmark = pytest.mark.skipif(
    not (_check_mlx() and sam31_available() and IMAGE.exists()),
    reason="SAM 3.1 weights or MLX runtime unavailable",
)

# zero objects, one object class (overlapping crowd), several prompts
PROMPTS = [["unicorn"], ["person"], ["person", "face"], ["face", "person", "unicorn"]]


@pytest.fixture(scope="module")
def image():
    return Image.open(IMAGE).convert("RGB")


def _assert_same(reference, fast, *, masks):
    assert len(reference) == len(fast)
    for ref, got in zip(reference, fast):
        assert ref.label == got.label
        assert ref.score == got.score
        assert ref.bbox_xyxy == got.bbox_xyxy
        if masks:
            assert got.mask is not None and np.array_equal(ref.mask, got.mask)
        else:
            assert got.mask is None


@pytest.mark.parametrize("task", ["detect", "segment"])
@pytest.mark.parametrize("prompts", PROMPTS)
def test_fast_path_matches_reference(image, prompts, task):
    kwargs = dict(threshold=0.15, resolution=504, task=task)
    reference = detect_multi(image, prompts, fast=False, **kwargs)
    fast = detect_multi(image, prompts, fast=True, **kwargs)
    _assert_same(reference, fast, masks=task == "segment")


def test_overlapping_crowd_exercises_suppression(image):
    reference = detect_multi(image, ["person"], threshold=0.05, resolution=504, task="segment", fast=False)
    fast = detect_multi(image, ["person"], threshold=0.05, resolution=504, task="segment", fast=True)
    assert len(reference) > 1
    _assert_same(reference, fast, masks=True)
