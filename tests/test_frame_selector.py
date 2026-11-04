"""Tests for visionbrain.frame_selector — FastScan scorer.

Covers the honest-measurement contracts:
- _select_sample_indices spans the whole video instead of only its start
- _build_quick_answer counts real detections (not label string length)
- inference failures are counted, never silently scored as "nothing there"
- negative answers qualify their sampled coverage

All inference is mocked; no MLX hardware or model weights are required.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from types import SimpleNamespace

from visionbrain.frame_selector import (
    FrameScore,
    FrameScores,
    _build_quick_answer,
    _cluster_regions,
    _select_sample_indices,
    score_frames,
)


def _mk_frame(
    idx: int,
    ts: float,
    score: float,
    dets: int = 0,
    label: str = "",
    failed: bool = False,
) -> FrameScore:
    """Build a FrameScore with has_query_match consistent with 0.2 threshold."""
    return FrameScore(
        frame_index=idx,
        timestamp=ts,
        relevance_score=score,
        detection_count=dets,
        top_label=label,
        has_query_match=score >= 0.2,
        failed=failed,
    )


# ──────────────────────────────────────────────────────────────────────────────
# _select_sample_indices
# ──────────────────────────────────────────────────────────────────────────────


class TestSelectSampleIndices:
    """Even-coverage sampling over the full video duration."""

    def test_long_video_spans_full_duration(self) -> None:
        # 30 minutes @ 30fps = 54000 frames; 5s interval -> 360 candidates
        idx = _select_sample_indices(54000, 30.0, 5.0, 60)
        candidates = list(range(0, 54000, 150))
        assert len(idx) == 60
        assert idx[0] == 0  # starts at the beginning
        assert idx[-1] == candidates[-1]  # reaches the very end (1795s of 1800s)
        assert idx == sorted(set(idx))
        assert set(idx).issubset(set(candidates))
        # Last sampled timestamp covers >= 90% of the video
        assert idx[-1] / 30.0 >= 0.9 * (54000 / 30.0)

    def test_short_video_passthrough_unchanged(self) -> None:
        # 30s @ 30fps -> 6 candidates, under max_frames: returned as-is
        idx = _select_sample_indices(900, 30.0, 5.0, 60)
        assert idx == list(range(0, 900, 150))

    def test_single_candidate_video(self) -> None:
        idx = _select_sample_indices(10, 30.0, 5.0, 60)
        assert idx == [0]

    def test_deterministic(self) -> None:
        a = _select_sample_indices(54000, 30.0, 5.0, 60)
        b = _select_sample_indices(54000, 30.0, 5.0, 60)
        assert a == b

    def test_zero_max_frames_returns_empty(self) -> None:
        assert _select_sample_indices(54000, 30.0, 5.0, 0) == []


# ──────────────────────────────────────────────────────────────────────────────
# _build_quick_answer
# ──────────────────────────────────────────────────────────────────────────────


class TestBuildQuickAnswer:
    """Quick-answer correctness: real counts, honest negatives."""

    def test_single_region_count_uses_detection_total(self) -> None:
        # 3 detections-frames in one region; label "truck" has length 5
        frames = [
            _mk_frame(0, 0.0, 0.5, dets=2, label="truck"),
            _mk_frame(1, 5.0, 0.6, dets=3, label="truck"),
            _mk_frame(2, 10.0, 0.5, dets=2, label="truck"),
        ]
        regions = _cluster_regions(frames, min_relevance=0.2)
        answer = _build_quick_answer(
            frames, regions, "truck", True, 15.0, 30.0, min_relevance=0.2
        )
        assert "7 total detections" in answer
        # The old bug reported len("truck") == 5 as the count
        assert "5 total detections" not in answer

    def test_multi_region_totals_all_regions(self) -> None:
        frames = [
            _mk_frame(0, 0.0, 0.5, dets=4, label="car"),
            _mk_frame(1, 5.0, 0.6, dets=2, label="car"),
            _mk_frame(2, 100.0, 0.5, dets=3, label="car"),  # gap > 10s
        ]
        regions = _cluster_regions(frames, min_relevance=0.2)
        answer = _build_quick_answer(
            frames, regions, "car", True, 120.0, 30.0, min_relevance=0.2
        )
        assert "2 regions" in answer
        assert "~9 total detections" in answer

    def test_min_relevance_respected_for_detection_totals(self) -> None:
        frames = [
            _mk_frame(0, 0.0, 0.25, dets=10, label="car"),
            _mk_frame(1, 5.0, 0.5, dets=3, label="car"),
        ]
        regions_strict = _cluster_regions(frames, min_relevance=0.3)
        answer_strict = _build_quick_answer(
            frames, regions_strict, "car", True, 15.0, 30.0, min_relevance=0.3
        )
        # 0.25 < 0.3, so only frame 1's 3 detections count
        assert "3 total detections" in answer_strict
        assert "10 total detections" not in answer_strict

        regions_loose = _cluster_regions(frames, min_relevance=0.2)
        answer_loose = _build_quick_answer(
            frames, regions_loose, "car", True, 15.0, 30.0, min_relevance=0.2
        )
        # With the lower threshold both frames count: 10 + 3
        assert "13 total detections" in answer_loose

    def test_not_relevant_states_sampled_coverage_and_failures(self) -> None:
        frames = [
            _mk_frame(i, i * (895.0 / 59.0), 0.0, failed=(i % 20 == 0))
            for i in range(60)
        ]
        answer = _build_quick_answer(
            frames,
            [],
            "trucks",
            False,
            900.0,
            30.0,
            min_relevance=0.2,
            frames_failed=3,
            sampled_start_s=0.0,
            sampled_end_s=895.0,
        )
        assert "No 'trucks' detected" in answer
        assert "60 sampled frames" in answer
        assert "0:00–14:55" in answer
        assert "15:00 total" in answer
        assert "(3 frames failed to score)" in answer

    def test_not_relevant_without_failures_omits_failure_note(self) -> None:
        frames = [_mk_frame(i, i * 5.0, 0.0) for i in range(10)]
        answer = _build_quick_answer(
            frames,
            [],
            "trucks",
            False,
            900.0,
            30.0,
            min_relevance=0.2,
            frames_failed=0,
            sampled_start_s=0.0,
            sampled_end_s=45.0,
        )
        assert "failed to score" not in answer
        assert "10 sampled frames" in answer

    def test_empty_frames_message(self) -> None:
        answer = _build_quick_answer([], [], "x", False, 10.0, 30.0)
        assert answer == "Could not read any frames from the video."


class TestFrameScoreContracts:
    """New fields must default so existing positional construction keeps working."""

    def test_frame_score_failed_defaults_false(self) -> None:
        f = FrameScore(0, 0.0, 0.5, 2, "car", True)
        assert f.failed is False

    def test_frame_scores_new_fields_default(self) -> None:
        fs = FrameScores(
            video_path="v.mp4",
            total_frames=10,
            fps=30.0,
            duration_s=0.33,
            frames_scored=1,
            is_relevant=False,
            quick_answer="",
        )
        assert fs.frames_failed == 0
        assert fs.sampled_span_s == 0.0
        d = fs.to_dict()
        assert d["frames_failed"] == 0
        assert d["sampled_span_s"] == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# score_frames (integration, mocked inference)
# ──────────────────────────────────────────────────────────────────────────────


def _write_clip(path, frames: int = 90, fps: float = 30.0, size=(64, 64)) -> bool:
    """Write a tiny synthetic MJPG .avi clip. Returns False if unsupported."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        return False
    for i in range(frames):
        val = int((i * 2) % 255)
        frame = np.full((size[1], size[0], 3), val, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    cap = cv2.VideoCapture(str(path))
    ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == frames
    cap.release()
    return ok


class TestScoreFrames:
    """score_frames with mocked Falcon inference (no weights required)."""

    def _mock_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "visionbrain.loader.falcon_perception_record",
            lambda: SimpleNamespace(can_load=True, note=""),
        )

    def test_mocked_success_counts_and_coverage(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = tmp_path / "clip.avi"
        if not _write_clip(clip):
            pytest.skip("cv2.VideoWriter could not write MJPG clips on this host")
        self._mock_ready(monkeypatch)

        def fake_detect(pil_img, query, max_new_tokens=100):
            return [SimpleNamespace(label="vehicle")] * 3, {}

        monkeypatch.setattr("visionbrain.fp_inference.detect", fake_detect)

        # 90 frames @ 30fps = 3s; 0.5s interval -> candidates [0,15,30,45,60,75]
        result = score_frames(str(clip), "vehicle", sample_every_n_seconds=0.5)

        assert result.frames_scored == 6
        assert result.frames_failed == 0
        assert result.is_relevant is True
        assert all(not f.failed for f in result.frame_scores)
        assert result.sampled_span_s == pytest.approx(75 / 30.0)
        # 6 frames x 3 detections = 18; single region -> exact total
        assert "18 total detections" in result.quick_answer
        d = result.to_dict()
        assert d["frames_failed"] == 0
        assert d["sampled_span_s"] == pytest.approx(2.5)

    def test_failed_frames_are_counted_and_flagged(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = tmp_path / "clip.avi"
        if not _write_clip(clip):
            pytest.skip("cv2.VideoWriter could not write MJPG clips on this host")
        self._mock_ready(monkeypatch)

        calls = {"n": 0}

        def flaky_detect(pil_img, query, max_new_tokens=100):
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise RuntimeError("simulated Falcon failure")
            return [SimpleNamespace(label="vehicle")] * 3, {}

        monkeypatch.setattr("visionbrain.fp_inference.detect", flaky_detect)

        result = score_frames(str(clip), "vehicle", sample_every_n_seconds=0.5)

        assert result.frames_scored == 6
        assert result.frames_failed == 3
        failed = [f for f in result.frame_scores if f.failed]
        assert len(failed) == 3
        assert all(f.relevance_score == 0.0 for f in failed)
        assert all(f.detection_count == 0 for f in failed)
        assert result.is_relevant is True  # 3 good frames still score 0.72
        d = result.to_dict()
        assert d["frames_failed"] == 3
        assert sum(1 for f in d["frame_scores"] if f["failed"]) == 3

    def test_missing_video_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Cannot open video"):
            score_frames("/nonexistent/nope.avi", "vehicle")

    def test_model_not_ready_raises(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clip = tmp_path / "clip.avi"
        if not _write_clip(clip):
            pytest.skip("cv2.VideoWriter could not write MJPG clips on this host")
        monkeypatch.setattr(
            "visionbrain.loader.falcon_perception_record",
            lambda: SimpleNamespace(can_load=False, note="weights not cached"),
        )
        with pytest.raises(RuntimeError, match="Falcon Perception not ready"):
            score_frames(str(clip), "vehicle")
