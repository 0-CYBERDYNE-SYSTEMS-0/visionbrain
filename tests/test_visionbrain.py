"""Pytest suite for VisionBrain.

Run with:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Ensure visionbrain is importable
VBRAIN = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(VBRAIN))

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Loader tests (no MLX required)
# ──────────────────────────────────────────────────────────────────────────────

class TestLoader:
    def test_check_mlx_handles_runtime_error(self, monkeypatch):
        import builtins
        from visionbrain import loader

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("mlx"):
                raise RuntimeError("[metal::load_device] No Metal device available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert loader._check_mlx() is False

    def test_falcon_perception_record(self):
        from visionbrain.loader import falcon_perception_record, HF_CACHE
        rec = falcon_perception_record()
        assert rec.hf_id == "tiiuae/Falcon-Perception"
        assert rec.cache_dir == HF_CACHE / "models--tiiuae--Falcon-Perception"
        # CI runners do not have local model weights; can_load is environment-dependent.
        print(f"\n  Falcon Perception: cached={rec.is_cached} ({rec.disk_gb} GB), can_load={rec.can_load}, note={rec.note}")

    def test_sam31_record(self):
        from visionbrain.loader import sam31_record, HF_CACHE
        rec = sam31_record()
        assert rec.hf_id == "mlx-community/sam3.1-bf16"
        assert rec.cache_dir == HF_CACHE / "models--mlx-community--sam3.1-bf16"
        # is_cached=True only if >0.5 GB downloaded
        print(f"\n  SAM 3.1: cached={rec.is_cached} ({rec.disk_gb} GB), can_load={rec.can_load}, note={rec.note}")

    def test_falcon_ocr_record(self):
        from visionbrain.loader import falcon_ocr_record, FALCON_OCR_HF_REPO, HF_CACHE
        rec = falcon_ocr_record()
        assert rec.hf_id == FALCON_OCR_HF_REPO == "tiiuae/Falcon-OCR"
        assert rec.cache_dir == HF_CACHE / "models--tiiuae--Falcon-OCR"
        # Registry-only entry: no MLX inference path for Falcon-OCR, ever.
        assert rec.can_load is False
        if rec.is_cached:
            assert "vLLM/CUDA" in rec.note
        else:
            # CI runners have no cached weights; the note must be actionable.
            assert rec.is_cached is False
            assert "huggingface-cli download tiiuae/Falcon-OCR" in rec.note
        print(f"\n  Falcon OCR: cached={rec.is_cached} ({rec.disk_gb} GB), can_load={rec.can_load}, note={rec.note}")

    def test_all_records_includes_falcon_ocr(self):
        from visionbrain import loader
        recs = loader.all_records()
        ids = [r.hf_id for r in recs]
        assert loader.FALCON_OCR_HF_REPO in ids
        assert loader.falcon_perception_record().hf_id in ids
        assert loader.falcon_perception_300m_record().hf_id in ids
        assert loader.sam31_record().hf_id in ids
        assert len(ids) == 5

    def test_falcon_perception_300m_record(self):
        from visionbrain.loader import (
            FALCON_PERCEPTION_300M_REPO,
            HF_CACHE,
            falcon_perception_300m_record,
        )
        rec = falcon_perception_300m_record()
        assert rec.hf_id == FALCON_PERCEPTION_300M_REPO == "tiiuae/Falcon-Perception-300M"
        assert rec.cache_dir == HF_CACHE / "models--tiiuae--Falcon-Perception-300M"
        # Detection-only variant: never emits masks regardless of cache state.
        print(f"\n  Falcon 300M: cached={rec.is_cached} ({rec.disk_gb} GB), can_load={rec.can_load}, note={rec.note}")

    def test_fp_model_id_env_override(self, monkeypatch):
        import visionbrain.fp_inference as fp

        monkeypatch.delenv("VB_FALCON_MODEL", raising=False)
        assert fp._selected_model_id("tiiuae/Falcon-Perception") == "tiiuae/Falcon-Perception"
        monkeypatch.setenv("VB_FALCON_MODEL", "tiiuae/Falcon-Perception-300M")
        assert fp._selected_model_id("tiiuae/Falcon-Perception") == "tiiuae/Falcon-Perception-300M"
        monkeypatch.setenv("VB_FALCON_MODEL", "   ")
        assert fp._selected_model_id("tiiuae/Falcon-Perception") == "tiiuae/Falcon-Perception"

    def test_falcon_repo_accessible(self):
        from visionbrain.loader import FALCON_REPO, falcon_repo
        if not FALCON_REPO.exists():
            pytest.skip(f"Falcon-Perception repo not available: {FALCON_REPO}")
        rec = falcon_repo()
        assert rec.exists(), f"Falcon-Perception repo should exist at {rec}"
        assert (rec / "falcon_perception").exists()

    def test_status_printer(self):
        from visionbrain.loader import print_status
        # Just verify it doesn't crash
        print_status()


# ──────────────────────────────────────────────────────────────────────────────
# Falcon Perception tests (requires mlx + cached weights)
# ──────────────────────────────────────────────────────────────────────────────

class TestFalconPerception:
    @pytest.fixture
    def test_image(self):
        from PIL import Image
        path = Path.home() / "Falcon-Perception" / "test_results" / "friends_people.jpg"
        if not path.exists():
            pytest.skip(f"Test image not found: {path}")
        return Image.open(path)

    def test_segment(self, test_image):
        from visionbrain.fp_inference import segment
        from visionbrain.loader import falcon_perception_record

        rec = falcon_perception_record()
        if not rec.can_load:
            pytest.skip(f"Falcon Perception not ready: {rec.note}")

        masks, stats = segment(test_image, "person", max_new_tokens=200)
        assert isinstance(masks, list)
        assert stats.total_ms > 0
        print(f"\n  Segment 'person': {len(masks)} masks in {stats.total_ms:.0f}ms")
        if masks:
            assert masks[0].mask_id == 1
            assert 0 <= masks[0].centroid_x <= 1
            assert 0 <= masks[0].centroid_y <= 1

    def test_detect(self, test_image):
        from visionbrain.fp_inference import detect
        from visionbrain.loader import falcon_perception_record

        rec = falcon_perception_record()
        if not rec.can_load:
            pytest.skip(f"Falcon Perception not ready: {rec.note}")

        detections, stats = detect(test_image, "person", max_new_tokens=200)
        assert isinstance(detections, list)
        assert stats.total_ms > 0
        print(f"\n  Detect 'person': {len(detections)} detections in {stats.total_ms:.0f}ms")
        if detections:
            assert 0 <= detections[0].cx <= 1
            assert 0 <= detections[0].cy <= 1

    def test_attribute_expression(self):
        from visionbrain.fp_inference import detect
        from visionbrain.loader import falcon_perception_record
        from PIL import Image
        import numpy as np

        rec = falcon_perception_record()
        if not rec.can_load:
            pytest.skip(f"Falcon Perception not ready: {rec.note}")

        # Create a synthetic test image with a simple shape
        img = Image.fromarray(
            np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        )
        detections, stats = detect(img, "red object", max_new_tokens=200)
        print(f"\n  Attribute 'red object': {len(detections)} detections in {stats.total_ms:.0f}ms")
        # Don't assert on count — synthetic image, just verify it runs

    def test_ocr(self):
        from visionbrain.fp_inference import ocr
        from visionbrain.loader import falcon_perception_record
        from PIL import Image

        rec = falcon_perception_record()
        if not rec.can_load:
            pytest.skip(f"Falcon Perception not ready: {rec.note}")

        path = Path.home() / "Falcon-Perception" / "test_results" / "vis_all_people_0.jpg"
        if not path.exists():
            pytest.skip("OCR test image not found")
        img = Image.open(path)
        detections, text, stats = ocr(img, "read all text", max_new_tokens=500)
        assert isinstance(text, str)
        print(f"\n  OCR: {len(detections)} text regions, {len(text)} chars markup, {stats.total_ms:.0f}ms")


# ──────────────────────────────────────────────────────────────────────────────
# Agent tools tests (no torch)
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentTools:
    def test_masks_to_vlm_json(self):
        from visionbrain.agent_tools import masks_to_vlm_json

        masks = {
            1: {
                "id": 1,
                "area_fraction": 0.05,
                "centroid_norm": {"x": 0.5, "y": 0.5},
                "bbox_norm": {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6},
                "image_region": "center",
                "rle": {"size": [100, 100], "counts": "..."},
            }
        }
        out = masks_to_vlm_json(masks)
        assert isinstance(out, list)
        assert out[0]["id"] == 1
        assert "rle" not in out[0]  # RLE should be stripped

    def test_compute_relations(self):
        from visionbrain.agent_tools import compute_relations
        from pycocotools import mask as mask_utils
        import numpy as np

        # Two overlapping masks
        arr1 = np.zeros((100, 100), dtype=np.uint8)
        arr1[20:60, 20:60] = 1
        arr2 = np.zeros((100, 100), dtype=np.uint8)
        arr2[40:80, 40:80] = 1

        rle1 = mask_utils.encode(np.asfortranarray(arr1))
        rle2 = mask_utils.encode(np.asfortranarray(arr2))

        def str_rle(r):
            out = dict(r)
            out["counts"] = out["counts"].decode("utf-8")
            return out

        masks = {
            1: {"id": 1, "area_fraction": 0.16, "centroid_norm": {"x": 0.4, "y": 0.4},
                "bbox_norm": {"x1": 0.2, "y1": 0.2, "x2": 0.6, "y2": 0.6},
                "image_region": "center-left", "rle": str_rle(rle1)},
            2: {"id": 2, "area_fraction": 0.16, "centroid_norm": {"x": 0.6, "y": 0.6},
                "bbox_norm": {"x1": 0.4, "y1": 0.4, "x2": 0.8, "y2": 0.8},
                "image_region": "center-right", "rle": str_rle(rle2)},
        }

        result = compute_relations(masks, [1, 2])
        assert "pairs" in result
        assert "1_vs_2" in result["pairs"]
        pair = result["pairs"]["1_vs_2"]
        assert "iou" in pair
        assert pair["1_left_of_2"] is True
        assert pair["1_above_2"] is True


# ──────────────────────────────────────────────────────────────────────────────
# Viz tests
# ──────────────────────────────────────────────────────────────────────────────

class TestViz:
    @pytest.fixture
    def sample_mask(self):
        from visionbrain.fp_inference import MaskResult
        from pycocotools import mask as mask_utils
        import numpy as np

        arr = np.zeros((100, 100), dtype=np.uint8)
        arr[20:60, 20:60] = 1
        rle = mask_utils.encode(np.asfortranarray(arr))
        rle_str = dict(rle)
        rle_str["counts"] = rle_str["counts"].decode("utf-8")

        return MaskResult(
            mask_id=1,
            centroid_x=0.4,
            centroid_y=0.4,
            bbox_x1=0.2,
            bbox_y1=0.2,
            bbox_x2=0.6,
            bbox_y2=0.6,
            area_fraction=0.16,
            image_region="center-left",
            rle=rle_str,
        )

    @pytest.fixture
    def sample_image(self):
        from PIL import Image
        import numpy as np
        return Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))

    def test_render_som(self, sample_image, sample_mask):
        from visionbrain.viz import render_som
        out = render_som(sample_image, [sample_mask])
        assert out.size == sample_image.size

    def test_render_detections(self, sample_image):
        from visionbrain.fp_inference import DetectionResult
        from visionbrain.viz import render_detections
        det = DetectionResult(
            label="cow", score=0.95,
            cx=0.5, cy=0.5, h=0.3, w=0.4,
        )
        out = render_detections(sample_image, [det])
        assert out.size == sample_image.size

    def test_get_crop(self, sample_image, sample_mask):
        from visionbrain.viz import get_crop
        crop = get_crop(sample_image, sample_mask, pad=0.1)
        assert crop.width < sample_image.width
        assert crop.height < sample_image.height

    def test_compute_relations_viz(self, sample_mask):
        from visionbrain.viz import compute_relations
        # Need at least 2 masks — create a second one
        from visionbrain.fp_inference import MaskResult
        from pycocotools import mask as mask_utils
        import numpy as np

        arr2 = np.zeros((100, 100), dtype=np.uint8)
        arr2[60:90, 60:90] = 1
        rle2 = mask_utils.encode(np.asfortranarray(arr2))
        rle2_str = dict(rle2)
        rle2_str["counts"] = rle2_str["counts"].decode("utf-8")

        mask2 = MaskResult(
            mask_id=2,
            centroid_x=0.75,
            centroid_y=0.75,
            bbox_x1=0.6,
            bbox_y1=0.6,
            bbox_x2=0.9,
            bbox_y2=0.9,
            area_fraction=0.09,
            image_region="bottom-right",
            rle=rle2_str,
        )
        result = compute_relations([sample_mask, mask2])
        assert "pairs" in result
        assert result["pairs"]["1_vs_2"]["1_left_of_2"] is True
        assert result["pairs"]["1_vs_2"]["1_above_2"] is True


class TestReviewOutputs:
    def test_display_result_does_not_reuse_latest_without_propagation(self):
        from visionbrain.sam3_inference import _display_result_for_frame

        latest = object()
        assert _display_result_for_frame(
            should_process=False,
            latest_result=latest,
            propagated_result=None,
        ) is None
        assert _display_result_for_frame(
            should_process=False,
            latest_result=latest,
            propagated_result="propagated",
        ) == "propagated"
        assert _display_result_for_frame(
            should_process=True,
            latest_result=latest,
            propagated_result=None,
        ) is latest

    def test_review_reel_duration_matches_analyzed_frames(self, tmp_path):
        import cv2
        import numpy as np
        from visionbrain.sam3_inference import render_review_outputs

        video_path = tmp_path / "source.mp4"
        fps = 5.0
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (64, 48),
        )
        for i in range(6):
            frame = np.full((48, 64, 3), i * 30, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        frame_data = [
            {
                "frame_index": 1,
                "timestamp": 0.2,
                "n_detections": 1,
                "detections": [{
                    "label": "roof",
                    "score": 0.9,
                    "track_id": 3,
                    "bbox_xyxy": [10, 10, 30, 30],
                }],
            },
            {
                "frame_index": 4,
                "timestamp": 0.8,
                "n_detections": 0,
                "detections": [],
            },
        ]
        reel_path = tmp_path / "review.mp4"
        still_dir = tmp_path / "stills"

        result = render_review_outputs(
            str(video_path),
            frame_data,
            review_reel_path=str(reel_path),
            hold_seconds=0.4,
            still_dir=str(still_dir),
            prompts=["roof damage"],
        )

        cap = cv2.VideoCapture(str(reel_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        assert result["video_frames_written"] == 4
        assert frame_count == 4
        assert len(list(still_dir.glob("*.jpg"))) == 2


# ──────────────────────────────────────────────────────────────────────────────
# CLI smoke tests
# ──────────────────────────────────────────────────────────────────────────────

class TestCLI:
    def test_status_command(self):
        import subprocess
        import sys
        # Run as module so relative imports resolve
        result = subprocess.run(
            [sys.executable, "-m", "visionbrain", "status"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "VisionBrain Model Status" in result.stdout

    def test_detect_help(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "visionbrain", "detect", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--image" in result.stdout

    def test_sam3_help(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "visionbrain", "sam3", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--image" in result.stdout
        assert "--prompts" in result.stdout

    def test_analyze_help_review_outputs(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "visionbrain", "analyze", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--review-reel-output" in result.stdout
        assert "--hold-seconds" in result.stdout
        assert "--still-dir" in result.stdout

    def test_analyze_help_lfm_ground(self):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "visionbrain", "analyze", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--lfm-ground" in result.stdout

    def test_pilot_eval_smoke(self):
        import argparse
        import subprocess
        import sys
        # `pilot-eval --help` prints usage without error (same pattern as above)
        result = subprocess.run(
            [sys.executable, "-m", "visionbrain", "pilot-eval", "--help"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).parent.parent / "src"),
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "--video" in result.stdout
        assert "--ground-truth" in result.stdout

        # Missing video exits cleanly with code 1 and an error message
        from visionbrain import cli
        args = argparse.Namespace(
            video="/nonexistent/pilot-eval-missing.mp4",
            ground_truth=None,
        )
        with pytest.raises(SystemExit) as excinfo:
            cli.cmd_pilot_eval(args)
        assert excinfo.value.code == 1


class TestWebApp:
    def test_job_result_json_endpoints(self, tmp_path):
        from fastapi.testclient import TestClient
        from visionbrain import web_app

        det = tmp_path / "detections.json"
        det.write_text('{"processed_frames": 1, "frames": []}')
        report = tmp_path / "report.txt"
        report.write_text("Roof repair candidates: none visible.")
        fast = tmp_path / "fast.json"
        fast.write_text('{"quick_answer": "No roof damage detected.", "regions": []}')

        jid = "testjob123"
        web_app._jobs[jid] = {
            "id": jid,
            "kind": "analyze",
            "status": "done",
            "results": {
                "json": str(det),
                "report": str(report),
                "fast_json": str(fast),
            },
            "output": [],
            "error": None,
        }

        client = TestClient(web_app.app)
        assert client.get(f"/api/job/{jid}/detections").json()["processed_frames"] == 1
        assert "Roof repair" in client.get(f"/api/job/{jid}/report").json()["text"]
        assert client.get(f"/api/job/{jid}/fast").json()["quick_answer"].startswith("No roof")

        web_app._jobs.pop(jid, None)

    def test_find_upload_skips_stills_dir(self, tmp_path, monkeypatch):
        from fastapi import HTTPException

        from visionbrain import web_app

        (tmp_path / "abc123.mp4").write_bytes(b"fake")
        (tmp_path / "abc123_stills").mkdir()          # analyze sidecar directory
        (tmp_path / "abc123_stills" / "f0.jpg").write_bytes(b"j")
        monkeypatch.setattr(web_app, "UPLOADS", tmp_path)

        assert web_app._find_upload("abc123") == tmp_path / "abc123.mp4"
        try:
            web_app._find_upload("zzzz")
            raise AssertionError("expected HTTPException for missing upload")
        except HTTPException as exc:
            assert exc.status_code == 404

    def test_ui_feedback_surfaces_present(self):
        """The always-on feedback surfaces ship in the served index.html.

        Every processing state must be visible somewhere: the film run panel,
        the still run panel (image jobs were previously silent in the stage),
        the live stage states with the feed-stall flag, and the ask/report
        in-flight slot (WEB_UI_REWORK_PLAN.md §6 honesty model).
        """
        from fastapi.testclient import TestClient

        from visionbrain import web_app

        client = TestClient(web_app.app)
        html = client.get("/").text
        for element_id in (
            "mission-run",                          # film run panel
            "inspect-run", "ir-task", "ir-phase",   # still run panel + cells
            "ir-elapsed", "ir-latest",
            "live-empty", "live-stall",             # live stage states + stall flag
            "live-stats",
            "btn-live-ask", "btn-live-report",      # shared ask/report slot
        ):
            assert f'id="{element_id}"' in html, f"missing feedback surface: #{element_id}"
        # §13.1 also prescribes an external-origins grep here; it lands with
        # the P0 shell rewrite — the current shell still preconnects to
        # Google Fonts and the fix (self-hosted static/fonts/) is scoped
        # there, not in the feedback layer.

    def test_live_simplification_client_pins(self):
        """SIMPLIFICATION_SPEC §1/§2/§6 client behavior, pinned in served HTML.

        - The ?live shortcut connects as a viewer only: no liveStartToggle
          call and no hardcoded stream/prompt fallbacks (empty storage starts
          no source and auto-fills nothing).
        - liveMasksChange re-sends prompts only in hub mode; local sends one
          set_task message.
        - The 100s ask timeout no longer re-enables the ask/report buttons.
        """
        from fastapi.testclient import TestClient

        from visionbrain import web_app

        client = TestClient(web_app.app)
        html = client.get("/").text

        # §1 — ?live bootstrap is a viewer, not an auto-start demo hook.
        bootstrap = html.split("has('live')", 1)[1].split("} catch", 1)[0]
        assert "liveStartToggle" not in bootstrap
        assert "liveEngineConnect" in bootstrap          # viewer attach only
        assert "localStorage.getItem('vb.liveUrl') ||" not in html
        assert "car truck bus person bicycle" not in html

        # §2 — local mask toggle sends set_task only; the hub keeps its
        # documented prompt re-send.
        masks = html.split("function liveMasksChange", 1)[1].split("livePromptsNorm", 1)[0]
        assert "set_task" in masks
        assert "LIVE.mode !== 'hub'" in masks

        # §6 — the late timeout must not re-enable the buttons while the
        # request slot is still occupied.
        tick = html.split("function liveAskTick", 1)[1].split("function liveAskEnd", 1)[0]
        assert "setAskButtons" not in tick
        assert "late" in tick


# ──────────────────────────────────────────────────────────────────────────────
# Prompt router tests (no MLX required — pure Python)
# ──────────────────────────────────────────────────────────────────────────────

class TestPromptRouter:
    def test_basic_partition(self):
        from visionbrain.prompt_router import route
        res = route("boats and people near the pier")
        assert "boats" in res.segment_targets
        assert "people" in res.segment_targets
        assert "pier" in res.segment_targets
        assert res.semantic_query == "boats and people near the pier"

    def test_multiword_phrase_keeps_verb_attachment(self):
        from visionbrain.prompt_router import route
        res = route("trucks blocking the north access road")
        assert res.segment_targets == ["trucks blocking", "north access road"]
        assert res.semantic_query == "trucks blocking the north access road"

    def test_adjective_phrase_stays_whole(self):
        from visionbrain.prompt_router import route
        res = route("yellow school bus")
        assert res.segment_targets == ["yellow school bus"]

    def test_open_vocab_pass_through(self):
        # Words like these were throttled by the old concrete-noun whitelist.
        from visionbrain.prompt_router import route
        res = route("kayak and buoy near the crane")
        assert res.segment_targets == ["kayak", "buoy", "crane"]

    def test_stopword_only_query_has_no_targets(self):
        from visionbrain.prompt_router import route
        q = "find all of that here"
        res = route(q)
        assert res.segment_targets == []
        assert res.semantic_query == res.original_query == q
        assert "No trackable phrases" in res.routed_from

    def test_content_phrase_passes_through_to_sam(self):
        # "suspicious activity" is content under open-vocab pass-through:
        # only stopword-only queries produce empty targets.
        from visionbrain.prompt_router import route
        res = route("find any suspicious activity")
        assert res.segment_targets == ["suspicious activity"]
        assert res.semantic_query == "find any suspicious activity"

    def test_cap_at_eight_targets(self):
        from visionbrain.prompt_router import route
        res = route("kayak and canoe and buoy and crane and barge and ferry "
                    "and tug and sailboat and trawler and dinghy and skiff and yacht")
        assert len(res.segment_targets) == 8
        assert res.segment_targets[0] == "kayak"

    def test_dedupe_case_insensitive_preserves_order(self):
        from visionbrain.prompt_router import route
        res = route("Boats and boats and BOATS near the dock")
        assert res.segment_targets == ["Boats", "dock"]

    def test_pure_numbers_dropped(self):
        from visionbrain.prompt_router import route
        res = route("3 trucks and 12 near the gate")
        # "3 trucks" is a content phrase; the standalone number "12" is dropped.
        assert res.segment_targets == ["3 trucks", "gate"]

    def test_routed_from_format(self):
        from visionbrain.prompt_router import route
        res = route("boats and people")
        assert res.routed_from == "SAM phrases: ['boats', 'people'] | Gemma: full query"

    def test_empty_query(self):
        from visionbrain.prompt_router import route
        res = route("")
        assert res.segment_targets == []
        assert res.semantic_query == ""
        assert res.original_query == ""

    def test_route_fallback_defaults(self):
        from visionbrain.prompt_router import route_fallback
        assert route_fallback("anything at all") == ["person", "vehicle", "building", "animal"]


# ──────────────────────────────────────────────────────────────────────────────
# Shared detection-core tests (no MLX required — pure Python)
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectionCore:
    def test_box_iou_identical_is_one(self):
        from visionbrain.detection_core import box_iou

        box = [0.1, 0.1, 0.5, 0.5]
        assert box_iou(box, box) == 1.0

    def test_box_iou_disjoint_is_zero(self):
        from visionbrain.detection_core import box_iou

        assert box_iou([0.0, 0.0, 0.1, 0.1], [0.8, 0.8, 0.9, 0.9]) == 0.0

    def test_box_iou_partial_overlap(self):
        from visionbrain.detection_core import box_iou

        iou = box_iou([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.5])
        assert abs(iou - 0.5) < 1e-6

    def test_labels_compatible_substring(self):
        from visionbrain.detection_core import labels_compatible

        assert labels_compatible("cow", "brown cow")
        assert labels_compatible("PERSON", "person")
        assert not labels_compatible("cow", "car")
        assert not labels_compatible("", "cow")

    def test_dedup_detections_merges_near_centers(self):
        from visionbrain.detection_core import dedup_detections

        items = [
            {"label": "cow", "box": [0.10, 0.10, 0.20, 0.20]},
            {"label": "cow", "box": [0.105, 0.10, 0.205, 0.20]},  # center within 0.01
            {"label": "cow", "box": [0.60, 0.60, 0.80, 0.80]},    # far away
        ]
        kept = dedup_detections(items, threshold=0.01)
        assert len(kept) == 2

    def test_dedup_disabled_on_nonpositive(self):
        from visionbrain.detection_core import dedup_detections

        items = [{"label": "x", "box": [0.1, 0.1, 0.2, 0.2]}] * 3
        assert len(dedup_detections(items, threshold=0)) == 3

    def test_track_manager_assigns_stable_ids(self):
        from visionbrain.detection_core import PersistentTrackManager

        mgr = PersistentTrackManager()
        first = mgr.assign(
            [{"label": "cow", "box": [0.1, 0.1, 0.3, 0.3], "score": 0.9}],
            frame_id=1,
            now_ms=1000,
        )
        second = mgr.assign(
            [{"label": "cow", "box": [0.11, 0.11, 0.31, 0.31], "score": 0.9}],
            frame_id=2,
            now_ms=1100,
        )
        assert first[0]["track_state"] == "new"
        assert second[0]["track_id"] == first[0]["track_id"]
        assert second[0]["track_state"] == "active"

    def test_track_manager_expires_after_ttl(self):
        from visionbrain.detection_core import PersistentTrackManager

        mgr = PersistentTrackManager(ttl_ms=100)
        a = mgr.assign([{"label": "cow", "box": [0.1, 0.1, 0.3, 0.3]}],
                       frame_id=1, now_ms=1000)
        b = mgr.assign([{"label": "cow", "box": [0.1, 0.1, 0.3, 0.3]}],
                       frame_id=2, now_ms=5000)
        assert b[0]["track_id"] != a[0]["track_id"]
        assert b[0]["track_state"] == "new"

    def test_merge_validate_hard_suppresses_solos(self):
        from visionbrain.detection_core import merge_validate

        sam = [{"label": "cow", "score": 0.9, "box": [0.1, 0.1, 0.4, 0.4]},
               {"label": "car", "score": 0.8, "box": [0.7, 0.7, 0.95, 0.95]}]
        fal = [{"label": "cow", "score": 0.7, "box": [0.12, 0.12, 0.42, 0.42]}]
        overlay, llm, stats = merge_validate(sam, fal, mode="hard")
        assert len(overlay) == 1 and overlay[0]["source"] == "agree"
        assert stats["suppressed"] == 1

    def test_merge_validate_soft_keeps_everything(self):
        from visionbrain.detection_core import merge_validate

        sam = [{"label": "cow", "score": 0.9, "box": [0.1, 0.1, 0.4, 0.4]}]
        fal = [{"label": "tractor", "score": 0.6, "box": [0.6, 0.6, 0.9, 0.9]}]
        overlay, llm, stats = merge_validate(sam, fal, mode="soft")
        assert len(overlay) == 2 and stats["agree"] == 0

    @staticmethod
    def _rect_mask(rows=(10, 30), cols=(20, 80), height=50, width=100):
        import numpy as np

        m = np.zeros((height, width), dtype=np.uint8)
        m[rows[0]:rows[1], cols[0]:cols[1]] = 1
        return m

    def test_mask_to_polygon_traces_outline_normalized(self):
        from visionbrain.detection_core import mask_to_polygon

        poly = mask_to_polygon(self._rect_mask(), width=100, height=50)
        assert poly is not None and len(poly) >= 4
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        assert min(xs) == 0.2 and max(xs) == 0.79   # cols 20..79 / 100
        assert min(ys) == 0.2 and max(ys) == 0.58   # rows 10..29 / 50
        # every point inside 0-1 and pairs
        assert all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in poly)

    def test_mask_to_polygon_empty_and_invalid_return_none(self):
        import numpy as np

        from visionbrain.detection_core import mask_to_polygon

        assert mask_to_polygon(np.zeros((50, 100), dtype=np.uint8), 100, 50) is None
        assert mask_to_polygon(None, 100, 50) is None
        assert mask_to_polygon(np.zeros((0, 10)), 10, 0) is None
        assert mask_to_polygon(np.zeros((2, 2, 2), dtype=np.uint8), 2, 2) is None

    def test_mask_to_polygon_accepts_plain_lists(self):
        from visionbrain.detection_core import mask_to_polygon

        mask = [[0, 0, 0], [0, 1, 1], [0, 1, 1]]
        poly = mask_to_polygon(mask, width=3, height=3)
        assert poly is not None and len(poly) >= 4
        assert max(x for x, _ in poly) == round(2 / 3, 4)

    def test_mask_to_polygon_respects_max_points(self):
        from visionbrain.detection_core import mask_to_polygon

        poly = mask_to_polygon(self._rect_mask(), 100, 50, max_points=12)
        assert poly is not None
        assert len(poly) <= 16  # 2 * (max_points//2 + 1)
        assert len(poly) >= 6


class TestCrosscheck:
    def test_falcon_to_dets_center_size_to_pixel_corners(self):
        from visionbrain.crosscheck import falcon_to_dets

        det = types.SimpleNamespace(
            label="cow", score=0.9, cx=0.5, cy=0.25, h=0.5, w=0.5
        )
        dets = falcon_to_dets([det], orig_w=200, orig_h=100)
        assert len(dets) == 1
        box = dets[0]["bbox_xyxy"]
        # corners: (0.5±0.25)*200, (0.25±0.25)*100
        assert abs(box[0] - 50.0) < 0.2
        assert abs(box[1] - 0.0) < 0.2
        assert abs(box[2] - 150.0) < 0.2
        assert abs(box[3] - 50.0) < 0.2
        assert dets[0]["label"] == "cow"
        assert dets[0]["score"] == 0.9

    def test_falcon_to_dets_accepts_dicts(self):
        from visionbrain.crosscheck import falcon_to_dets

        dets = falcon_to_dets(
            [{"label": "car", "score": 0.5, "cx": 0.5, "cy": 0.5, "h": 1.0, "w": 1.0}],
            orig_w=100, orig_h=100,
        )
        assert dets[0]["bbox_xyxy"] == [0.0, 0.0, 100.0, 100.0]

    def test_crosscheck_perfect_match_agreement_one(self):
        from visionbrain.crosscheck import crosscheck

        sam = [{"label": "cow", "score": 0.9, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0], "track_id": 3}]
        fal = [{"label": "cow", "score": 0.8, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0]}]
        res = crosscheck(sam, fal)
        assert res.matched == 1
        assert res.sam_only == 0 and res.falcon_only == 0
        assert res.agreement == 1.0
        assert res.matches[0]["sam_track_id"] == 3
        assert res.matches[0]["iou"] == 1.0

    def test_crosscheck_disjoint_boxes_match_nothing(self):
        from visionbrain.crosscheck import crosscheck

        sam = [{"label": "cow", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
        fal = [{"label": "cow", "score": 0.8, "bbox_xyxy": [100.0, 100.0, 120.0, 120.0]}]
        res = crosscheck(sam, fal)
        assert res.matched == 0
        assert res.agreement == 0.0
        assert res.sam_only == 1 and res.falcon_only == 1
        assert res.matches == []

    def test_crosscheck_overlap_below_threshold_no_match(self):
        from visionbrain.crosscheck import crosscheck

        # IoU = 0.4 / 1.0 = 0.4 < 0.5
        sam = [{"label": "cow", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
        fal = [{"label": "cow", "score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 4.0]}]
        res = crosscheck(sam, fal)
        assert res.matched == 0
        assert res.agreement == 0.0

    def test_crosscheck_iou_at_threshold_matches(self):
        from visionbrain.crosscheck import crosscheck

        # IoU = 0.5 / 1.0 = 0.5 → >= threshold matches
        sam = [{"label": "cow", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
        fal = [{"label": "cow", "score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 5.0]}]
        res = crosscheck(sam, fal)
        assert res.matched == 1

    def test_crosscheck_greedy_one_to_one_highest_iou_first(self):
        from visionbrain.crosscheck import crosscheck

        sam = [
            {"label": "a", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},   # IoU 0.95 with falcon
            {"label": "b", "score": 0.8, "bbox_xyxy": [1.0, 0.0, 11.0, 10.0]},   # IoU ~0.77 with falcon
        ]
        fal = [{"label": "a", "score": 0.7, "bbox_xyxy": [0.0, 0.0, 9.5, 10.0]}]
        res = crosscheck(sam, fal)
        assert res.matched == 1
        assert res.sam_only == 1 and res.falcon_only == 0
        # Higher-IoU pair wins; one Falcon det can match only one SAM det.
        assert res.matches[0]["sam_label"] == "a"
        assert res.agreement == 0.5  # 1 / max(1, max(2, 1))

    def test_crosscheck_label_passthrough(self):
        from visionbrain.crosscheck import crosscheck

        sam = [{"label": "vehicle", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
        fal = [{"label": "truck", "score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
        res = crosscheck(sam, fal)
        assert res.matched == 1
        assert res.matches[0]["sam_label"] == "vehicle"
        assert res.matches[0]["falcon_label"] == "truck"

    def test_crosscheck_track_id_propagation(self):
        from visionbrain.crosscheck import crosscheck

        sam = [
            {"label": "cow", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "track_id": 7},
            {"label": "cow", "score": 0.9, "bbox_xyxy": [50.0, 0.0, 60.0, 10.0]},  # no track_id
        ]
        fal = [
            {"label": "cow", "score": 0.8, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]},
            {"label": "cow", "score": 0.8, "bbox_xyxy": [50.0, 0.0, 60.0, 10.0]},
        ]
        res = crosscheck(sam, fal)
        assert [m["sam_track_id"] for m in res.matches] == [7, None]

    def test_crosscheck_frame_index_defaults_unset(self):
        from visionbrain.crosscheck import crosscheck

        res = crosscheck([], [])
        assert res.frame_index == -1
        res.frame_index = 42  # CLI sets it after matching
        assert res.frame_index == 42

    def test_summarize_aggregates_and_means(self):
        from visionbrain.crosscheck import CrosscheckResult, summarize

        results = [
            CrosscheckResult(frame_index=1, matched=2, sam_only=0, falcon_only=0, agreement=1.0),
            CrosscheckResult(frame_index=2, matched=1, sam_only=1, falcon_only=0, agreement=0.5),
        ]
        agg = summarize(results)
        assert agg["frames"] == 2
        assert agg["matched"] == 3
        assert agg["sam_only"] == 1
        assert agg["falcon_only"] == 0
        assert abs(agg["agreement"] - 0.75) < 1e-9

    def test_summarize_empty_is_zero(self):
        from visionbrain.crosscheck import summarize

        agg = summarize([])
        assert agg == {"frames": 0, "matched": 0, "sam_only": 0, "falcon_only": 0, "agreement": 0.0}


class TestGrounding:
    # ── build_grounding_prompt ────────────────────────────────────────────────

    def test_prompt_lists_targets_and_canonical_format(self):
        from visionbrain.grounding import build_grounding_prompt

        prompt = build_grounding_prompt(["boat", "swimmer"])
        assert "boat" in prompt
        assert "swimmer" in prompt
        assert "<box>x1,y1,x2,y2</box>" in prompt
        assert "NONE" in prompt
        # The example line shows the canonical integer 0-1000 format.
        assert "<box>120,340,480,760</box>" in prompt

    def test_prompt_ignores_blank_targets(self):
        from visionbrain.grounding import build_grounding_prompt

        prompt = build_grounding_prompt(["", "  ", None])  # type: ignore[list-item]
        assert "objects" in prompt  # neutral fallback wording

    # ── parse_grounding_boxes: canonical <box> tags ───────────────────────────

    def test_parse_canonical_zero_to_1000_ints(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("<box>100,200,500,800</box> boat", 500, 250)
        assert len(boxes) == 1
        assert boxes[0]["label"] == "boat"
        x1, y1, x2, y2 = boxes[0]["bbox_xyxy"]
        assert x1 == pytest.approx(100 / 1000 * 500)
        assert y1 == pytest.approx(200 / 1000 * 250)
        assert x2 == pytest.approx(500 / 1000 * 500)
        assert y2 == pytest.approx(800 / 1000 * 250)

    def test_parse_zero_to_one_floats(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("<box>0.1,0.2,0.5,0.8</box> car", 400, 200)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]["bbox_xyxy"]
        assert x1 == pytest.approx(40.0)
        assert y1 == pytest.approx(40.0)
        assert x2 == pytest.approx(200.0)
        assert y2 == pytest.approx(160.0)

    def test_parse_zero_to_100_values(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("<box>10,20,50,80</box> person", 200, 100)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]["bbox_xyxy"]
        assert x1 == pytest.approx(20.0)
        assert y1 == pytest.approx(20.0)
        assert x2 == pytest.approx(100.0)
        assert y2 == pytest.approx(80.0)

    def test_parse_optional_spaces_and_float_values(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("<box> 100 , 200.5 , 500 , 800 </box>", 1000, 1000)
        assert len(boxes) == 1
        assert boxes[0]["bbox_xyxy"] == [100.0, 200.5, 500.0, 800.0]

    def test_parse_tag_case_insensitive(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("<BOX>100,200,500,800</BOX> boat", 1000, 1000)
        assert len(boxes) == 1
        assert boxes[0]["label"] == "boat"

    def test_parse_label_stripped_and_empty_ok(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("<box>100,200,500,800</box>  red sailboat ", 1000, 1000)
        assert boxes[0]["label"] == "red sailboat"
        boxes = parse_grounding_boxes("<box>100,200,500,800</box>", 1000, 1000)
        assert boxes[0]["label"] == ""

    def test_parse_multiple_boxes_with_labels(self):
        from visionbrain.grounding import parse_grounding_boxes

        text = "<box>100,200,500,800</box> boat\n<box>600,100,900,400</box> kayak"
        boxes = parse_grounding_boxes(text, 1000, 1000)
        assert [b["label"] for b in boxes] == ["boat", "kayak"]
        assert boxes[1]["bbox_xyxy"][0] == pytest.approx(600.0)

    def test_parse_unsorted_swapped_corners_ordered(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("<box>500,800,100,200</box> boat", 500, 250)
        x1, y1, x2, y2 = boxes[0]["bbox_xyxy"]
        assert x1 < x2 and y1 < y2
        assert x1 == pytest.approx(50.0) and x2 == pytest.approx(250.0)
        assert y1 == pytest.approx(50.0) and y2 == pytest.approx(200.0)

    def test_parse_clamps_out_of_range_to_image_bounds(self):
        from visionbrain.grounding import parse_grounding_boxes

        # x1,y1,x2,y2 = -50,1200,600,900 on a 0-1000 scale over 500x500:
        # raw pixels (-25, 600, 300, 450) → clamp → (0, 500, 300, 450) → order.
        boxes = parse_grounding_boxes("<box>-50,1200,600,900</box> wreck", 500, 500)
        x1, y1, x2, y2 = boxes[0]["bbox_xyxy"]
        assert x1 == 0.0  # -25 clamped to left edge
        assert x2 == pytest.approx(300.0)
        assert y1 == pytest.approx(450.0)
        assert y2 == 500.0  # 600 clamped to bottom edge, then ordered below 450
        assert x1 < x2 and y1 < y2

    # ── parse_grounding_boxes: fallbacks and empty replies ────────────────────

    def test_parse_parenthesized_fallback(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("(120,340),(480,760) boat", 500, 500)
        assert len(boxes) == 1
        x1, y1, x2, y2 = boxes[0]["bbox_xyxy"]
        assert x1 == pytest.approx(60.0)
        assert y1 == pytest.approx(170.0)
        assert x2 == pytest.approx(240.0)
        assert y2 == pytest.approx(380.0)
        assert boxes[0]["label"] == ""  # no tag → no label

    def test_parse_json_nested_array_fallback(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("[[100,200,500,800],[20,40,60,80]]", 500, 250)
        assert len(boxes) == 2
        # First box is 0-1000 scale, second is 0-100 scale (heuristic per box).
        assert boxes[0]["bbox_xyxy"] == pytest.approx([50.0, 50.0, 250.0, 200.0])
        assert boxes[1]["bbox_xyxy"] == pytest.approx([100.0, 100.0, 300.0, 200.0])

    def test_parse_json_flat_array_fallback(self):
        from visionbrain.grounding import parse_grounding_boxes

        boxes = parse_grounding_boxes("Here: [100,200,500,800] as requested", 1000, 1000)
        assert len(boxes) == 1
        assert boxes[0]["bbox_xyxy"] == [100.0, 200.0, 500.0, 800.0]

    def test_parse_json_pair_arrays_are_not_boxes(self):
        from visionbrain.grounding import parse_grounding_boxes

        # Corners as coordinate pairs must not be double-parsed as boxes.
        boxes = parse_grounding_boxes("[[120,340],[480,760]]", 500, 500)
        assert boxes == []

    def test_parse_none_reply_returns_empty(self):
        from visionbrain.grounding import parse_grounding_boxes

        assert parse_grounding_boxes("NONE", 500, 500) == []
        assert parse_grounding_boxes("none.", 500, 500) == []
        assert parse_grounding_boxes("", 500, 500) == []
        assert parse_grounding_boxes("   \n  ", 500, 500) == []

    def test_parse_prose_without_boxes_returns_empty(self):
        from visionbrain.grounding import parse_grounding_boxes

        assert parse_grounding_boxes("I cannot see any boats in this image.", 500, 500) == []

    # ── grounding_crosscheck ──────────────────────────────────────────────────

    def test_grounding_crosscheck_perfect_match(self):
        from visionbrain.grounding import grounding_crosscheck

        sam = [{"label": "boat", "score": 0.9, "bbox_xyxy": [10.0, 10.0, 50.0, 50.0], "track_id": 3}]
        boxes = [{"bbox_xyxy": [10.0, 10.0, 50.0, 50.0], "label": "boat"}]
        res = grounding_crosscheck(sam, boxes)
        assert res.matched == 1
        assert res.sam_only == 0 and res.falcon_only == 0
        assert res.agreement == 1.0
        assert res.matches[0]["iou"] == 1.0
        assert res.matches[0]["sam_track_id"] == 3

    def test_grounding_crosscheck_loose_default_accepts_coarse_box(self):
        from visionbrain.crosscheck import crosscheck
        from visionbrain.grounding import grounding_crosscheck

        # Coarse VLM box: IoU = 0.4 — too loose for the Falcon threshold (0.5),
        # matched by the grounding default (0.3).
        sam = [{"label": "boat", "score": 0.9, "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
        boxes = [{"bbox_xyxy": [0.0, 0.0, 10.0, 4.0], "label": "boat"}]
        assert grounding_crosscheck(sam, boxes).matched == 1
        assert crosscheck(sam, boxes, iou_threshold=0.5).matched == 0

    def test_grounding_crosscheck_empty_inputs(self):
        from visionbrain.grounding import grounding_crosscheck

        res = grounding_crosscheck([], [])
        assert res.matched == 0
        assert res.agreement == 0.0
        assert res.frame_index == -1  # caller sets it when the frame is known


class TestModelHost:
    def test_acquire_loads_once_and_refcounts(self):
        from visionbrain.model_host import ModelHost

        host = ModelHost()
        calls = []

        def loader():
            calls.append(1)
            return {"weights": True}

        first = host.acquire("k", loader)
        second = host.acquire("k", loader)
        assert first is second and len(calls) == 1
        host.release("k")
        assert host.resident() == ["k"]  # one holder remains
        host.release("k")
        assert host.resident() == []

    def test_release_unknown_key_is_noop(self):
        from visionbrain.model_host import ModelHost

        host = ModelHost()
        host.release("never-loaded")  # must not raise
        assert host.resident() == []

    def test_slow_load_does_not_block_other_keys(self):
        import threading

        from visionbrain.model_host import ModelHost

        host = ModelHost()
        proceed = threading.Event()
        load_started = threading.Event()

        def slow_loader():
            load_started.set()
            assert proceed.wait(5.0)
            return "B"

        t = threading.Thread(
            target=host.acquire, args=("b", slow_loader), daemon=True
        )
        t.start()
        assert load_started.wait(2.0)
        # While "b" is mid-load (outside the global lock) other keys stay
        # fully live — the old implementation stalled every acquire/release
        # behind a multi-second load.
        assert host.acquire("a", lambda: "A") == "A"
        host.release("a")
        proceed.set()
        t.join(5.0)
        assert host.acquire("b", lambda: "B") == "B"
        host.release("b")   # main thread's hold
        host.release("b")   # the loader thread's hold from acquire()
        assert host.resident() == []

    def test_same_key_concurrent_acquire_loads_once(self):
        import threading
        import time

        from visionbrain.model_host import ModelHost

        host = ModelHost()
        calls = []
        gate = threading.Event()

        def loader():
            calls.append(1)
            assert gate.wait(5.0)
            return "W"

        results = []

        def acquire():
            results.append(host.acquire("k", loader))

        threads = [threading.Thread(target=acquire, daemon=True) for _ in range(3)]
        for t in threads:
            t.start()
        time.sleep(0.2)  # let all three park (one loading, two waiting)
        gate.set()
        for t in threads:
            t.join(5.0)
        assert len(calls) == 1
        assert results == ["W", "W", "W"]
        for _ in range(3):
            host.release("k")
        assert host.resident() == []


class TestSupervisionBridge:
    """Compact-mask cache: round-trip, eviction, no id-reuse contamination."""

    @staticmethod
    def _detections_with_mask(second_row_active=False):
        import numpy as np
        import supervision as sv

        mask = np.zeros((2, 50, 100), dtype=bool)
        mask[0, 10:40, 20:80] = True
        if second_row_active:
            mask[1, 5:15, 5:15] = True
        return sv.Detections(
            xyxy=np.array([[20.0, 10.0, 80.0, 40.0],
                           [5.0, 5.0, 15.0, 15.0]], dtype=np.float32),
            mask=mask,
        )

    def test_to_compact_roundtrip(self):
        from visionbrain import supervision_bridge as sb

        dets = self._detections_with_mask()
        compact = sb.to_compact(dets)
        assert compact.mask is None
        restored = sb.from_compact(compact)
        assert restored.mask is not None
        assert restored.mask.shape == (2, 50, 100)
        assert bool(restored.mask[0, 10:40, 20:80].all())

    def test_cache_entry_evicted_when_detections_collected(self):
        import gc

        from visionbrain import supervision_bridge as sb

        compact = sb.to_compact(self._detections_with_mask())
        key = id(compact)
        assert key in sb._compact_mask_cache
        del compact
        gc.collect()
        # The weakref finalizer must evict — an unbounded id()-keyed cache
        # leaks every converted mask and risks recycled-id contamination.
        assert key not in sb._compact_mask_cache
        assert key not in sb._compact_shape_cache

    def test_from_compact_no_cross_contamination(self):
        import numpy as np

        from visionbrain import supervision_bridge as sb

        first_mask = self._detections_with_mask().mask
        second_mask = self._detections_with_mask(second_row_active=True).mask
        sb.to_compact(self._detections_with_mask())
        second_compact = sb.to_compact(
            self._detections_with_mask(second_row_active=True)
        )
        restored = sb.from_compact(second_compact)
        # Must be SECOND's masks — a recycled-id cache would risk handing
        # back the first object's masks under the new boxes.
        assert np.array_equal(restored.mask, second_mask)
        assert not np.array_equal(restored.mask, first_mask)

    def test_from_compact_without_cache_entry_is_identity(self):
        import numpy as np
        import supervision as sv

        from visionbrain import supervision_bridge as sb

        dets = sv.Detections(
            xyxy=np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
        )
        assert sb.from_compact(dets) is dets


class TestVLMRegistry:
    def test_lfm3b_registered_with_expected_checkpoint(self):
        from visionbrain.vlm_registry import MODELS

        assert MODELS["lfm3b"] == "LiquidAI/LFM2.5-VL-3B-MLX-4bit"
        assert MODELS["lfm"] == "LiquidAI/LFM2.5-VL-450M-MLX-4bit"
        assert MODELS["gemma"] == "mlx-community/gemma-4-e2b-it-4bit"

    def test_set_model_roundtrip_and_reject(self):
        from visionbrain.vlm_registry import current_key, set_model

        try:
            assert set_model("lfm3b") == "LiquidAI/LFM2.5-VL-3B-MLX-4bit"
            assert current_key() == "lfm3b"
            with pytest.raises(ValueError):
                set_model("does-not-exist")
        finally:
            set_model("gemma")

    def test_position_label_bands(self):
        from visionbrain.vlm_registry import position_label

        assert position_label(0.5, 0.5) == "center of frame"
        assert position_label(0.1, 0.9) == "lower left of frame"
        assert position_label(0.9, 0.1) == "upper right of frame"

    def test_format_detection_lines(self):
        from visionbrain.vlm_registry import format_detection_lines

        out = format_detection_lines(
            [{"label": "cow", "score": 0.87,
              "centroid_norm": {"x": 0.2, "y": 0.2}, "source": "sam"}]
        )
        assert out == "- cow, 87% confidence, upper left of frame [sam]"


class TestVLMSettings:
    """Custom OpenAI-compatible backend settings store (CI-safe — no network)."""

    def test_save_load_roundtrip(self, tmp_path):
        import json
        from visionbrain import gemma_inference as gi

        p = tmp_path / "settings.json"
        saved = gi.save_vlm_settings(
            base_url="http://localhost:1234/v1",
            model="test-model",
            api_key="sk-secret",
            path=p,
        )
        # save() redacts the key in its return value...
        assert saved == {"base_url": "http://localhost:1234/v1", "model": "test-model", "api_key": ""}
        # ...but the key material persists on disk, in a 0600 file
        assert json.loads(p.read_text())["api_key"] == "sk-secret"
        assert (p.stat().st_mode & 0o777) == 0o600

        loaded = gi.load_vlm_settings(path=p)
        assert loaded == {
            "base_url": "http://localhost:1234/v1",
            "model": "test-model",
            "api_key": "sk-secret",
        }

    def test_missing_or_corrupt_file_yields_empty_settings(self, tmp_path):
        from visionbrain import gemma_inference as gi

        assert gi.load_vlm_settings(path=tmp_path / "absent.json") == {
            "base_url": "", "model": "", "api_key": "",
        }
        corrupt = tmp_path / "corrupt.json"
        corrupt.write_text("{not valid json")
        assert gi.load_vlm_settings(path=corrupt) == {
            "base_url": "", "model": "", "api_key": "",
        }

    def test_save_empty_preserves_and_clear_key_wipes_only_key(self, tmp_path):
        from visionbrain import gemma_inference as gi

        p = tmp_path / "settings.json"
        gi.save_vlm_settings(base_url="http://x/v1", model="m1", api_key="k1", path=p)

        saved = gi.save_vlm_settings(path=p)  # all-empty: nothing overwritten
        assert saved["base_url"] == "http://x/v1"
        assert saved["model"] == "m1"
        assert gi.load_vlm_settings(path=p)["api_key"] == "k1"

        saved = gi.save_vlm_settings(model="m2", clear_key=True, path=p)
        assert saved["model"] == "m2"
        assert gi.load_vlm_settings(path=p) == {
            "base_url": "http://x/v1", "model": "m2", "api_key": "",
        }

    def test_custom_backend_configured_requires_base_url_and_model(self, tmp_path, monkeypatch):
        from visionbrain import gemma_inference as gi

        p = tmp_path / "settings.json"
        monkeypatch.setattr(gi, "settings_path", lambda: p)

        assert gi.custom_backend_configured() is False  # no file yet
        gi.save_vlm_settings(base_url="http://x/v1", path=p)
        assert gi.custom_backend_configured() is False  # base_url only
        gi.save_vlm_settings(model="m", path=p)
        assert gi.custom_backend_configured() is True
        gi.save_vlm_settings(clear_key=True, path=p)
        assert gi.custom_backend_configured() is True   # api_key is irrelevant

    def test_available_backend_custom_first(self, monkeypatch):
        from visionbrain import gemma_inference as gi

        def no_network(*args, **kwargs):
            raise AssertionError("network probe attempted")

        monkeypatch.setattr(gi.urllib.request, "urlopen", no_network)
        monkeypatch.setattr(gi, "custom_backend_configured", lambda: True)
        assert gi.available_backend() == "custom"

        monkeypatch.setattr(gi, "custom_backend_configured", lambda: False)
        assert gi.available_backend() != "custom"

    def test_custom_chat_request_shape(self, monkeypatch):
        import json
        from visionbrain import gemma_inference as gi

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({
                    "choices": [{"message": {"content": "answer from custom"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 7},
                }).encode("utf-8")

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.headers)
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return FakeResponse()

        monkeypatch.setattr(gi.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(gi, "load_vlm_settings", lambda path=None: {
            "base_url": "http://example.test:1234/v1", "model": "test-model", "api_key": ""})

        text, raw, latency_s = gi._custom_chat(
            [{"role": "user", "content": "hi"}], max_tokens=64, temperature=0.1
        )
        assert text == "answer from custom"
        assert raw["usage"]["completion_tokens"] == 7
        assert latency_s >= 0.0
        assert captured["url"] == "http://example.test:1234/v1/chat/completions"
        assert "Authorization" not in captured["headers"]  # no key → no header
        assert captured["payload"]["model"] == "test-model"
        assert captured["payload"]["max_tokens"] == 64
        assert captured["payload"]["temperature"] == 0.1
        assert captured["payload"]["messages"] == [{"role": "user", "content": "hi"}]

        # With a saved api key, a Bearer Authorization header is sent
        monkeypatch.setattr(gi, "load_vlm_settings", lambda path=None: {
            "base_url": "http://example.test:1234/v1/", "model": "test-model",
            "api_key": "sk-test"})
        captured.clear()
        gi._custom_chat([{"role": "user", "content": "hi"}], max_tokens=64, temperature=0.1)
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        assert captured["url"] == "http://example.test:1234/v1/chat/completions"  # trailing / stripped

    def test_custom_chat_unconfigured_or_malformed_raises(self, monkeypatch, tmp_path):
        import json
        from visionbrain import gemma_inference as gi

        monkeypatch.setattr(gi, "load_vlm_settings", lambda path=None: {
            "base_url": "", "model": "", "api_key": ""})
        with pytest.raises(RuntimeError, match="not configured"):
            gi._custom_chat([{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.1)

        class BadResponse:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps({"choices": "nope"}).encode("utf-8")

        monkeypatch.setattr(gi.urllib.request, "urlopen", lambda req, timeout=None: BadResponse())
        monkeypatch.setattr(gi, "load_vlm_settings", lambda path=None: {
            "base_url": "http://example.test", "model": "m", "api_key": ""})
        with pytest.raises(RuntimeError, match="Malformed response"):
            gi._custom_chat([{"role": "user", "content": "hi"}], max_tokens=8, temperature=0.1)


class TestMlxCompat:
    """mlx_vlm load shims (gemma-4 quantized ScaledLinear, lfm2_vl layernorm)."""

    def test_apply_all_idempotent(self):
        from visionbrain.mlx_compat import apply_all

        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            pytest.skip("mlx_vlm not installed")
        apply_all()
        apply_all()  # second call must be a no-op, not a double patch

    def test_scaled_linear_quantized_matches_reference(self):
        try:
            import mlx.core as mx
            from mlx_vlm.models.gemma4.language import ScaledLinear
        except ImportError:
            pytest.skip("mlx_vlm gemma4 arch not available")

        from visionbrain.mlx_compat import ensure_scaled_linear_quantization

        ensure_scaled_linear_quantization()
        if not hasattr(ScaledLinear, "to_quantized"):
            pytest.fail("ScaledLinear.to_quantized shim was not installed")

        layer = ScaledLinear(128, 64, scalar=0.25)
        layer.weight = mx.random.normal((64, 128))
        x = mx.random.normal((2, 16, 128))

        ql = layer.to_quantized(group_size=64, bits=4, mode="affine")
        got = ql(x)
        w, scales, biases = mx.quantize(layer.weight, 64, 4, mode="affine")
        want = (x @ mx.dequantize(w, scales, biases, 64).T) * 0.25
        assert mx.abs(got - want).max() < 1e-4

    def test_lfm_guard_corrects_only_proven_layernorm(self, tmp_path):
        import json

        try:
            import mlx_vlm  # noqa: F401
        except ImportError:
            pytest.skip("mlx_vlm not installed")

        from visionbrain.mlx_compat import ensure_lfm_projector_layernorm
        from mlx_vlm.utils import load_config

        ensure_lfm_projector_layernorm()

        def make_case(name, layernorm_in_weights, declared):
            d = tmp_path / name
            d.mkdir()
            (d / "config.json").write_text(json.dumps({
                "model_type": "lfm2-vl",
                "projector_use_layernorm": declared,
            }))
            weight_map = {"language_model.model.embed_tokens.weight": "m.safetensors"}
            if layernorm_in_weights:
                weight_map["multi_modal_projector.layer_norm.weight"] = "m.safetensors"
            (d / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map})
            )
            return d

        lying = make_case("lying", layernorm_in_weights=True, declared=False)
        assert load_config(lying)["projector_use_layernorm"] is True

        honest = make_case("honest", layernorm_in_weights=False, declared=False)
        assert load_config(honest)["projector_use_layernorm"] is False


class TestLiveTracking:
    """LiveSamTracker with fully injected inference — no mlx_vlm needed."""

    @staticmethod
    def _make_tracker(detect_every=2, backbone_every=15):
        from visionbrain import live_tracking as lt

        class FakeResult:
            scores = [0.91, 0.55]
            boxes = [[10.0, 10.0, 50.0, 50.0], [100.0, 20.0, 160.0, 70.0]]
            labels = ["plane", "truck"]
            track_ids = [1, 2]
            masks = None

        class FakeTracker:
            def update(self, result):
                return result

        detect_calls: list[int] = []
        backbone_calls: list[int] = []

        def fake_detect(predictor, backbone_features, prompts, image_size,
                        threshold, encoder_cache=None):
            detect_calls.append(1)
            return FakeResult()

        def fake_backbone(model, pixel_values):
            backbone_calls.append(1)
            return object()

        def fake_preprocess(processor, image):
            return "pixels"

        # Pre-seed the module cache so _ensure_loaded never touches mlx_vlm.
        lt._loaded[(lt.DEFAULT_MODEL, lt.DEFAULT_RESOLUTION)] = {
            "model": object(), "processor": object(), "predictor": object(),
        }
        tracker = lt.LiveSamTracker(
            detect_every=detect_every,
            backbone_every=backbone_every,
            backbone_fn=fake_backbone,
            detect_fn=fake_detect,
            preprocess_fn=fake_preprocess,
            tracker=FakeTracker(),
        )
        return tracker, detect_calls, backbone_calls

    def teardown_method(self):
        from visionbrain import live_tracking as lt

        lt._loaded.clear()

    def test_step_detect_then_held_republish(self):
        tracker, det_calls, _bb = self._make_tracker()

        items = tracker.step("img", ["plane"], "detect", 200, 200,
                             frame_id=1, timestamp_ms=1000)
        assert len(items) == 2
        assert items[0]["source"] == "sam"
        assert items[0]["box"][0] == 0.05  # 10/200 normalized
        assert items[0]["track_state"] == "active"

        held = tracker.step("img", ["plane"], "detect", 200, 200,
                            frame_id=2, timestamp_ms=1400)
        assert held[0]["track_state"] == "predicted"
        assert held[0]["stale_ms"] == 400
        assert held[0]["track_id"] == items[0]["track_id"]

    def test_backbone_cache_respects_backbone_every(self):
        tracker, _det, bb_calls = self._make_tracker(backbone_every=2)
        for f in range(1, 7):  # detect ticks 0,2,4 → backbone recomputes on each
            tracker.step("img", ["plane"], "detect", 200, 200,
                         frame_id=f, timestamp_ms=f * 100)
        assert len(bb_calls) == 3

    def test_empty_scene_keeps_detect_every_throttle(self):
        """A detect that finds nothing must not force a full detect every
        frame — the held path runs with an empty set too."""
        from visionbrain import live_tracking as lt

        class EmptyResult:
            scores = []
            boxes = []
            labels = []
            track_ids = []
            masks = None

        class FakeTracker:
            def update(self, result):
                return result

        detect_calls = []

        def fake_detect(predictor, backbone_features, prompts, image_size,
                        threshold, encoder_cache=None):
            detect_calls.append(1)
            return EmptyResult()

        lt._loaded[(lt.DEFAULT_MODEL, lt.DEFAULT_RESOLUTION)] = {
            "model": object(), "processor": object(), "predictor": object(),
        }
        tracker = lt.LiveSamTracker(
            detect_every=3,
            backbone_fn=lambda m, p: object(),
            detect_fn=fake_detect,
            preprocess_fn=lambda proc, img: "pixels",
            tracker=FakeTracker(),
        )
        for f in range(1, 7):  # detects on frames 1 and 4 only
            items = tracker.step("img", ["plane"], "detect", 200, 200,
                                 frame_id=f, timestamp_ms=f * 100)
            assert items == []
        assert len(detect_calls) == 2

    def test_ensure_loaded_reapplies_threshold_on_cache_hit(self):
        from visionbrain import live_tracking as lt

        class FakePredictor:
            score_threshold = 0.15

        pred = FakePredictor()
        lt._loaded[("m", 1008)] = {
            "model": object(), "processor": object(), "predictor": pred,
        }
        try:
            _m, _p, got = lt._ensure_loaded("m", 1008, 0.4)
            assert got is pred
            assert pred.score_threshold == 0.4
        finally:
            lt._loaded.clear()

    def test_step_segment_emits_polygon_by_default(self):
        import numpy as np

        from visionbrain import live_tracking as lt

        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[10:50, 10:50] = 1

        class FakeResult:
            scores = [0.9]
            boxes = [[10.0, 10.0, 50.0, 50.0]]
            labels = ["boat"]
            track_ids = [7]
            masks = [mask]

        class FakeTracker:
            def update(self, result):
                return result

        lt._loaded[(lt.DEFAULT_MODEL, lt.DEFAULT_RESOLUTION)] = {
            "model": object(), "processor": object(), "predictor": object(),
        }
        tracker = lt.LiveSamTracker(
            backbone_fn=lambda m, p: object(),
            detect_fn=lambda *a, **k: FakeResult(),
            preprocess_fn=lambda proc, img: "pixels",
            tracker=FakeTracker(),
        )
        items = tracker.step("img", ["boat"], "segment", 200, 200,
                             frame_id=1, timestamp_ms=1000)
        assert len(items) == 1
        poly = items[0].get("polygon")
        assert poly is not None and len(poly) >= 4
        xs = [p[0] for p in poly]
        assert min(xs) == 0.05 and max(xs) == round(49 / 200, 4)

    def test_step_detect_task_stays_box_only(self):
        import numpy as np

        from visionbrain import live_tracking as lt

        class FakeResult:
            scores = [0.9]
            boxes = [[10.0, 10.0, 50.0, 50.0]]
            labels = ["boat"]
            track_ids = [7]
            masks = [np.ones((200, 200), dtype=np.uint8)]

        class FakeTracker:
            def update(self, result):
                return result

        lt._loaded[(lt.DEFAULT_MODEL, lt.DEFAULT_RESOLUTION)] = {
            "model": object(), "processor": object(), "predictor": object(),
        }
        tracker = lt.LiveSamTracker(
            backbone_fn=lambda m, p: object(),
            detect_fn=lambda *a, **k: FakeResult(),
            preprocess_fn=lambda proc, img: "pixels",
            tracker=FakeTracker(),
        )
        items = tracker.step("img", ["boat"], "detect", 200, 200,
                             frame_id=1, timestamp_ms=1000)
        assert "polygon" not in items[0]

    def test_make_item_carries_polygon(self):
        from visionbrain.live_engine import make_item

        item = make_item(
            [10.0, 10.0, 50.0, 50.0], 100, 100, "boat", 0.9, 3, "east",
            polygon=[[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]],
        )
        assert item["polygon"] == [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]
        # out-of-range points clamp into 0-1
        item_clamped = make_item(
            [0.0, 0.0, 100.0, 100.0], 100, 100, "boat", 0.9, 3, "east",
            polygon=[[-0.2, 0.1], [0.5, 1.7], [0.5, 0.5]],
        )
        assert item_clamped["polygon"][0] == [0.0, 0.1]
        assert item_clamped["polygon"][1] == [0.5, 1.0]
        # falsy polygon omitted entirely
        assert "polygon" not in make_item(
            [0, 0, 1, 1], 1, 1, "x", 1.0, 0, "east", polygon=None
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
