"""Tests for the pilot evaluation harness (no MLX, no real inference).

score_fn and frame_reader are injected, so no video decoding or model
weights are needed. cv2 is never imported at module top — frame_selector
is only imported for its dataclasses (a declared dependency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure visionbrain is importable
VBRAIN = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(VBRAIN))

import pytest
from PIL import Image

from visionbrain.frame_selector import FrameScore, FrameScores
from visionbrain.pilot_eval import PilotReport, run_pilot_eval, validate_ground_truth


# ──────────────────────────────────────────────────────────────────────────────
# Helpers — hand-built FrameScores (setattr covers fields arriving in parallel)
# ──────────────────────────────────────────────────────────────────────────────

def make_frame(index, ts, *, rel=1.0, dets=1, label="person", match=True,
               failed=False):
    """Build a FrameScore, optionally marked failed."""
    f = FrameScore(
        frame_index=index, timestamp=ts, relevance_score=rel,
        detection_count=dets, top_label=label, has_query_match=match,
    )
    f.failed = failed
    return f


def make_scores(frames, *, duration_s=305.0, frames_scored=None,
                frames_failed=0, sampled_span_s=300.0):
    """Build a FrameScores around the given FrameScore list."""
    n = frames_scored if frames_scored is not None else len(frames)
    fs = FrameScores(
        video_path="video.mp4", total_frames=9150, fps=30.0,
        duration_s=duration_s, frames_scored=n, is_relevant=True,
        quick_answer="quick answer", frame_scores=list(frames),
    )
    fs.frames_failed = frames_failed
    fs.sampled_span_s = sampled_span_s
    return fs


def noop_reader(video_path, frame_index):
    """Frame reader that never produces an image."""
    return None


def image_reader(video_path, frame_index):
    """Frame reader returning a tiny in-memory image (no decode)."""
    return Image.new("RGB", (4, 3), color="red")


# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth validation
# ──────────────────────────────────────────────────────────────────────────────

class TestGroundTruthValidation:
    def test_valid_ground_truth_normalizes(self):
        result = validate_ground_truth({
            "video": "clip.mp4",
            "query": "person",
            "events": [
                {"label": "person", "start_s": 12, "end_s": 20.0},
                {"label": "truck", "start_s": 30.5, "end_s": 40.5,
                 "min_count": 3},
            ],
        })
        assert result["query"] == "person"
        assert result["events"][0] == {
            "label": "person", "start_s": 12.0, "end_s": 20.0, "min_count": 1}
        assert result["events"][1]["min_count"] == 3

    def test_empty_events_list_is_valid(self):
        result = validate_ground_truth({"query": "person", "events": []})
        assert result["events"] == []

    @pytest.mark.parametrize("bad,match", [
        (["not", "an", "object"], "JSON object"),
        ({"events": []}, "non-empty 'query'"),
        ({"query": "   ", "events": []}, "non-empty 'query'"),
        ({"query": "person"}, "'events' list"),
        ({"query": "person", "events": {"a": 1}}, "'events' must be a list"),
        ({"query": "person", "events": ["nope"]},
         r"events\[0\] must be a JSON object"),
        ({"query": "person", "events": [{"label": "", "start_s": 1, "end_s": 2}]},
         "non-empty 'label'"),
        ({"query": "person", "events": [{"label": "x", "start_s": "12", "end_s": 20}]},
         "'start_s' must be a number"),
        ({"query": "person", "events": [{"label": "x", "start_s": 12, "end_s": None}]},
         "'end_s' must be a number"),
        ({"query": "person", "events": [{"label": "x", "start_s": 20.0, "end_s": 12.0}]},
         r"start_s \(20\.0\) must be <= end_s \(12\.0\)"),
        ({"query": "person", "events": [{"label": "x", "start_s": 1, "end_s": 2,
                                         "min_count": 0}]},
         "'min_count' must be an integer >= 1"),
        ({"query": "person", "events": [{"label": "x", "start_s": 1, "end_s": 2,
                                         "min_count": "2"}]},
         "'min_count' must be an integer >= 1"),
    ])
    def test_malformed_ground_truth_raises_valueerror(self, bad, match):
        with pytest.raises(ValueError, match=match):
            validate_ground_truth(bad)


# ──────────────────────────────────────────────────────────────────────────────
# run_pilot_eval
# ──────────────────────────────────────────────────────────────────────────────

class TestRunPilotEval:
    def test_all_events_detected_latency_math(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        scores = make_scores([
            make_frame(315, 10.5, dets=1),    # inside tolerance window
            make_frame(396, 13.2, dets=2),
            make_frame(480, 16.0, dets=1),
        ])
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)

        assert report.events_total == 1
        assert report.events_detected == 1
        assert report.events_missed == 0
        assert report.missed_events == []
        # First matching frame is 10.5s -> latency -1.5s (tolerance) kept raw
        ev = report.per_event[0]
        assert ev["detected"] is True
        assert ev["latency_s"] == pytest.approx(-1.5)
        assert len(ev["support"]) == 3
        assert report.false_alert_count == 0

    def test_missed_event_is_listed_not_spun(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        scores = make_scores([make_frame(3000, 100.0, dets=3)])
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)

        assert report.events_detected == 0
        assert report.events_missed == 1
        assert report.missed_events == [
            {"label": "person", "start_s": 12.0, "end_s": 20.0}]
        assert report.per_event[0]["detected"] is False
        assert report.per_event[0]["latency_s"] is None
        assert report.per_event[0]["support"] == []
        # The lone detection is outside the window -> false alert
        assert report.false_alert_count == 1
        assert report.false_alerts[0]["timestamp"] == pytest.approx(100.0)

    def test_false_alert_tolerance_edges_are_frame_level(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 10.0, "end_s": 20.0}]}
        # tolerance 3.0 -> window [7.0, 23.0]; edges are inclusive
        scores = make_scores([
            make_frame(195, 6.5, dets=1),    # just outside -> false alert
            make_frame(210, 7.0, dets=1),    # exactly at edge -> match
            make_frame(690, 23.0, dets=1),   # exactly at edge -> match
            make_frame(705, 23.5, dets=1),   # just outside -> false alert
        ])
        report = run_pilot_eval("video.mp4", gt, tolerance_s=3.0,
                                score_fn=lambda *a, **k: scores)

        assert report.events_detected == 1
        timestamps = [fa["timestamp"] for fa in report.false_alerts]
        assert timestamps == [pytest.approx(6.5), pytest.approx(23.5)]
        assert report.false_alert_count == 2
        assert len(report.per_event[0]["support"]) == 2

    def test_min_count_not_met_is_missed_not_false_alert(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0,
                          "min_count": 2}]}
        scores = make_scores([make_frame(450, 15.0, dets=1)])
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)

        assert report.events_missed == 1
        assert report.per_event[0]["detected"] is False
        # Inside the window, so it is not a false alert — it caused the miss
        assert report.false_alert_count == 0

    def test_below_min_relevance_frames_are_not_detections(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        scores = make_scores([make_frame(450, 15.0, rel=0.1, dets=1)])
        report = run_pilot_eval("video.mp4", gt, min_relevance=0.2,
                                score_fn=lambda *a, **k: scores)

        assert report.events_missed == 1
        assert report.false_alert_count == 0  # not a detection at all

    def test_failed_frames_excluded_and_surfaced(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        scores = make_scores(
            [
                make_frame(450, 15.0, dets=5, rel=1.0, failed=True),  # excluded
                make_frame(451, 15.5, dets=1, rel=0.9),
            ],
            frames_scored=5,
            frames_failed=1,
        )
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)

        # The failed frame must not count as a detection despite dets=5
        assert len(report.per_event[0]["support"]) == 1
        assert report.per_event[0]["support"][0]["frame_index"] == 451
        assert report.coverage["frames_failed"] == 1
        assert report.coverage["frames_scored"] == 5
        assert any("1 of 5 sampled frames failed" in c for c in report.caveats)

    def test_empty_events_all_detections_become_false_alerts(self):
        gt = {"query": "person", "events": []}
        scores = make_scores([
            make_frame(300, 10.0, dets=1),
            make_frame(600, 20.0, dets=2),
            make_frame(900, 30.0, dets=1),
        ])
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)

        assert report.events_total == 0
        assert report.events_detected == 0
        assert report.false_alert_count == 3
        assert len(report.false_alerts) == 3
        assert "0/0 events detected" in report.summary()[2]

    def test_score_fn_receives_sampling_params_and_query(self):
        captured = {}

        def spy_score_fn(video_path, query, **kwargs):
            captured["video_path"] = video_path
            captured["query"] = query
            captured.update(kwargs)
            return make_scores([make_frame(450, 15.0, dets=1)])

        gt = {"query": "vehicles at the gate",
              "events": [{"label": "truck", "start_s": 14.0, "end_s": 16.0}]}
        run_pilot_eval(
            "video.mp4", gt,
            score_fn=spy_score_fn,
            sample_every_n_seconds=2.5,
            max_frames=42,
            resolution=480,
            min_relevance=0.3,
        )
        assert captured["video_path"] == "video.mp4"
        assert captured["query"] == "vehicles at the gate"
        assert captured["sample_every_n_seconds"] == 2.5
        assert captured["max_frames"] == 42
        assert captured["resolution"] == 480
        assert captured["min_relevance"] == 0.3

    def test_coverage_dict_and_caveats_always_present(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        # Full coverage + no failed frames -> only the two unconditional caveats
        scores = make_scores([make_frame(450, 15.0, dets=1)],
                             sampled_span_s=305.0)
        report = run_pilot_eval("video.mp4", gt, sample_every_n_seconds=5.0,
                                max_frames=60,
                                score_fn=lambda *a, **k: scores)

        assert report.coverage == {
            "duration_s": pytest.approx(305.0),
            "sampled_span_s": pytest.approx(305.0),
            "frames_scored": 1,
            "frames_failed": 0,
            "sample_every_n_seconds": 5.0,
            "max_frames": 60,
        }
        # Caveats 1 and 2 are unconditional
        assert any("sampled every 5" in c and "sample interval" in c
                   for c in report.caveats)
        assert any("label agreement is not independent" in c
                   for c in report.caveats)
        # No failure / partial-coverage caveat when everything is clean
        assert len(report.caveats) == 2

    def test_partial_coverage_adds_caveat(self):
        gt = {"query": "person", "events": []}
        scores = make_scores([], duration_s=305.0, sampled_span_s=240.0)
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)
        assert any("Coverage was partial: sampled span 240.0s of 305.0s" in c
                   for c in report.caveats)


# ──────────────────────────────────────────────────────────────────────────────
# Evidence
# ──────────────────────────────────────────────────────────────────────────────

class TestEvidence:
    def test_evidence_file_written_when_reader_supplies_image(self, tmp_path):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        scores = make_scores([make_frame(450, 15.0, dets=1)])
        evidence_dir = tmp_path / "evidence"
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores,
                                evidence_dir=str(evidence_dir),
                                frame_reader=image_reader)

        ev = report.per_event[0]
        assert ev["evidence_saved"] is True
        assert ev["evidence"] == str(evidence_dir / "event00_person_15.0s.jpg")
        assert Path(ev["evidence"]).exists()
        with Image.open(ev["evidence"]) as img:
            assert img.size == (4, 3)

    def test_evidence_saved_false_when_reader_returns_none(self, tmp_path):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        scores = make_scores([make_frame(450, 15.0, dets=1)])
        evidence_dir = tmp_path / "evidence"
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores,
                                evidence_dir=str(evidence_dir),
                                frame_reader=noop_reader)

        ev = report.per_event[0]
        assert ev["evidence_saved"] is False
        assert ev["evidence"] is None
        assert list(evidence_dir.glob("*.jpg")) == []

    def test_no_evidence_dir_means_no_evidence_fields_set(self):
        gt = {"query": "person",
              "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0}]}
        scores = make_scores([make_frame(450, 15.0, dets=1)])
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)
        assert report.per_event[0]["evidence_saved"] is False
        assert report.per_event[0]["evidence"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Report serialization
# ──────────────────────────────────────────────────────────────────────────────

class TestPilotReport:
    def test_to_dict_save_roundtrip(self, tmp_path):
        gt = {"query": "person",
              "events": [
                  {"label": "person", "start_s": 12.0, "end_s": 20.0},
                  {"label": "truck", "start_s": 100.0, "end_s": 110.0},
              ]}
        scores = make_scores(
            [make_frame(450, 15.0, dets=2, rel=0.8),
             make_frame(4500, 150.0, dets=1)],  # outside both windows
            frames_scored=4, frames_failed=1,
        )
        report = run_pilot_eval("video.mp4", gt, score_fn=lambda *a, **k: scores)
        assert isinstance(report, PilotReport)

        out = tmp_path / "report.json"
        report.save(out)
        loaded = json.loads(out.read_text())
        assert loaded == report.to_dict()

        assert loaded["events_total"] == 2
        assert loaded["events_detected"] == 1
        assert loaded["events_missed"] == 1
        assert loaded["false_alert_count"] == 1
        assert loaded["coverage"]["frames_failed"] == 1
        assert isinstance(loaded["runtime_s"], float)
        assert loaded["runtime_s"] >= 0.0
        assert len(loaded["caveats"]) >= 3  # sampling + labels + 1 failed frame

    def test_summary_lines_are_honest(self):
        report = PilotReport(
            video="video.mp4", query="person",
            events_total=5, events_detected=3, events_missed=2,
            missed_events=[{"label": "person", "start_s": 12.0, "end_s": 20.0}],
            false_alert_count=4,
            false_alerts=[{"timestamp": 30.0}],
            per_event=[
                {"detected": True, "latency_s": 1.2},
                {"detected": True, "latency_s": 6.0},
                {"detected": False, "latency_s": None},
            ],
            coverage={"duration_s": 305.0, "sampled_span_s": 300.0,
                      "frames_scored": 61, "frames_failed": 2,
                      "sample_every_n_seconds": 5.0, "max_frames": 60},
            runtime_s=12.5,
            caveats=["sampled", "labels", "failures"],
        )
        lines = report.summary()
        text = "\n".join(lines)
        assert "3/5 events detected, 2 missed; 4 false alerts" in text
        assert "per-event latency 1.2-6.0s" in text
        assert "MISSED: person @ 12.0-20.0s" in text
        assert "300/305s sampled" in text
        assert "2 frames failed" in text
        assert "caveat: sampled" in text

    def test_summary_without_detections_has_no_latency_range(self):
        report = PilotReport(
            video="video.mp4", query="person",
            events_total=1, events_detected=0, events_missed=1,
            false_alert_count=0,
            per_event=[{"detected": False, "latency_s": None}],
        )
        text = "\n".join(report.summary())
        assert "0/1 events detected, 1 missed; 0 false alerts" in text
        assert "per-event latency" not in text
