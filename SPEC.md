# VisionBrain — Technical Specification

> Agricultural vision AI on Apple Silicon. Powered by Falcon Perception (MLX) and SAM 3.1.

---

## Overview

VisionBrain is a Python library and CLI that provides farmer-friendly access to two state-of-the-art vision models, running entirely locally on Apple Silicon via MLX:

- **Falcon Perception** (tiiuae/Falcon-Perception) — 3B-param vision-language model for segmentation, detection, and OCR
- **SAM 3.1** (facebook/sam3.1) — Meta's Segment Anything Model 3.1 with Object Multiplex for video tracking

**Design principle:** zero modifications to any existing project. VisionBrain reads from cached weights and the Falcon-Perception git repo, imports from them, and never writes back.

---

## Repository Layout

```
VisionBrain/
├── SPEC.md                   ← this file
├── README.md                 ← user-facing docs
├── pyproject.toml            ← package metadata
├── src/
│   └── visionbrain/
│       ├── __init__.py
│       ├── __main__.py        ← python -m visionbrain entry point
│       ├── loader.py         ← model registry, cache status, availability
│       ├── fp_inference.py   ← Falcon Perception: segment(), detect(), ocr()
│       ├── sam3_inference.py ← SAM 3.1: detect_multi(), track_video(), track_realtime()
│       ├── viz.py            ← Set-of-Marks rendering, crop extraction, relations
│       ├── agent_tools.py    ← agent-facing: ground_expression(), compute_relations()
│       ├── agent_loop.py     ← VLM agent: tool loop, context pruning
│       ├── cli.py            ← CLI commands
│       └── references/
│           └── system_prompt.txt
├── tests/
│   └── test_visionbrain.py   ← 16 tests (all pass)
└── assets/
    └── samples/              ← test images and output
```

---

## Module Specifications

### `loader.py` — Model Registry

**Public API:**
- `falcon_perception_record() -> ModelRecord` — cache path, size, availability
- `sam31_record() -> ModelRecord` — cache path, size, availability
- `print_status()` — human-readable table to stdout
- `falcon_repo() -> Path` — path to Falcon-Perception git repo (read-only)
- `sam31_cache_path() -> Path | None` — SAM 3.1 cache, or None if not downloaded

**Key behavior:**
- Falcon Perception requires weights cached at `~/.cache/huggingface/hub/models--tiiuae--Falcon-Perception`
- SAM 3.1 requires weights at `~/.cache/huggingface/hub/models--facebook--sam3.1` (download via `huggingface-cli download facebook/sam3.1`)
- Both require `mlx` and `mlx_vlm` importable in the Python environment

---

### `fp_inference.py` — Falcon Perception Pipeline

**Public API:**
- `segment(image, expression, *, ...) -> (list[MaskResult], InferenceStats)`
- `detect(image, expression, *, ...) -> (list[DetectionResult], InferenceStats)`
- `ocr(image, question, *, ...) -> (list[DetectionResult], str, InferenceStats)`

**MaskResult fields:** `mask_id`, `centroid_x/y`, `bbox_x1/y1/x2/y2`, `area_fraction`, `image_region`, `rle`

**DetectionResult fields:** `label`, `score`, `cx`, `cy`, `h`, `w`

**InferenceStats fields:** `preprocess_ms`, `generation_ms`, `total_ms`, `prefill_tokens`, `decoded_tokens`, `tokens_per_sec`, `n_masks`, `n_detections`

**Model caching:** The MLX model is loaded once per process and cached in `_model_cache`. Subsequent calls are fast.

**OCR output format:** The Falcon Perception OCR task emits special markup tokens (`<|coord|>`, `<|size|>`, `<|seg|>`) rather than readable text. The tokenizer preserves these with `skip_special_tokens=False`. Consumers decode to get the raw token stream.

**Key implementation notes:**
- Uses `falcon_perception.mlx.batch_inference.process_batch_and_generate()` for preprocessing
- Uses `BatchInferenceEngine.generate()` for token generation
- RLE masks are resized from inference resolution to original image resolution using `pycocotools`
- Centroid and bbox coordinates are normalized 0–1

---

### `sam3_inference.py` — SAM 3.1 Wrapper

**Public API:**
- `sam31_available() -> bool`
- `detect_multi(image, prompts, *, ...) -> list[Sam31Detection]`
- `track_video(video_path, prompts, output_path, *, ...) -> VideoTrackStats`
- `track_realtime(camera_or_video, prompts, *, ...) -> None`

**Requirements:** SAM 3.1 weights cached via `huggingface-cli download facebook/sam3.1`

**Key optimizations:**
- `detect_multi`: Runs vision backbone ONCE, then DETR per prompt. Cost = 1x ViT + Nx text/DETR
- `track_video`: Skips ViT on intermediate frames via backbone caching (~67ms saved/frame)
- Only re-runs DETR every `detect_every` frames; tracker propagates via mask decoder between detections

---

### `viz.py` — Visualization

**Public API:**
- `render_som(image, masks, *, ...) -> PIL.Image` — colored numbered mask overlay
- `render_detections(image, detections, *, ...) -> PIL.Image` — bounding boxes
- `get_crop(image, mask, *, pad=0.05) -> PIL.Image` — tight crop around a mask
- `compute_relations(masks) -> dict` — pairwise IoU, left/right, above/below, centroid distance

---

### `agent_tools.py` — Agent Tools

Drops in as a replacement for `falcon_perception/demo/agent/fp_tools.py` using the MLX path.

**Public API:**
- `run_ground_expression(image, expression, *, ...) -> dict[int, dict]` — same return type as `fp_tools.run_ground_expression()`
- `compute_relations(masks, mask_ids) -> dict`
- `masks_to_vlm_json(masks) -> list[dict]`

**Key behavior:** Resizes RLE masks from inference resolution to original image resolution before returning metadata.

---

### `agent_loop.py` — VLM Agent

**Public API:**
- `VLMClient(api_key, model, base_url)` — OpenAI-compatible client (subclass for others)
- `run_agent(image, question, client, *, ...) -> AgentResult`

**AgentResult fields:** `answer`, `supporting_mask_ids`, `final_image`, `history`, `n_fp_calls`, `n_vlm_calls`

**Tool set:** `ground_expression`, `get_crop`, `compute_relations`, `answer`

**Context pruning:** Keeps system prompt + original user message + last tool-call round + last N messages. Never exceeds ~4 rounds.

**Context window management:** Image is sent via base64 in the first user message only; subsequent turns pass metadata JSON. Total context per turn stays under ~3000 tokens.

---

### `cli.py` — CLI Commands

| Command | Description |
|---------|-------------|
| `visionbrain status` | Print model cache status |
| `visionbrain detect` | Bounding-box detection (fast) |
| `visionbrain segment` | Pixel-accurate segmentation (SoM output) |
| `visionbrain ocr` | Text reading from images |
| `visionbrain sam3` | SAM 3.1 multi-prompt detection |
| `visionbrain track` | SAM 3.1 video object tracking |
| `visionbrain agent` | VLM-powered visual reasoning |

---

## One-Time Setup

```bash
# SAM 3.1 weights (requires HF account + Meta approval)
huggingface-cli download facebook/sam3.1
```

---

## Test Coverage

```
16 tests — all pass
├── TestLoader (4)       — cache status, repo access
├── TestFalconPerception (4) — segment, detect, attribute expr, OCR
├── TestAgentTools (2)   — masks_to_vlm_json, compute_relations
├── TestViz (4)         — SoM, detections, crop, relations
└── TestCLI (2)          — status command, detect --help
```

Run: `FALCON_PY=~/Library/Caches/pypoetry/virtualenvs/falcon-perception-NVnkjaN--py3.12/bin/python`
`$FALCON_PY -m pytest tests/ -v`

---

## Dependencies

Core (must have):
- `mlx`
- `mlx_vlm`
- `transformers`
- `pillow`
- `pycocotools`
- `numpy`
- `opencv-python` (SAM video tracking)

Dev:
- `pytest`
