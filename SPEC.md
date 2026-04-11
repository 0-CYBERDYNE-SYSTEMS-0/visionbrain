# VisionBrain — Technical Specification

> Agricultural vision AI on Apple Silicon. Two-machine pipeline: SAM 3.1 runs locally, Gemma 4 26B runs on remote GPU server.

---

## Architecture

**Two-machine design** — because Gemma 4 26B requires ~32GB and this Mac Mini has 16GB.

```
THIS MAC MINI (100.72.41.118, Mac Mini M4 16GB)
  SAM 3.1 (mlx-community/sam3.1-bf16) — local MLX
  └── Annotated video MP4
  └── Per-frame detection JSON (track IDs, centroids, bboxes, area fractions)
           │
           │  HTTP POST to remote
           ▼
REMOTE GPU SERVER (100.72.41.118:8080, rapid-mlx)
  Gemma 4 26B (mlx-community/gemma-4-26b-a4b-it-4bit)
  └── Field reports, Q&A, anomaly detection
```

---

## Overview

VisionBrain is a Python library and CLI providing farmer-friendly access to three state-of-the-art vision models, running entirely locally on Apple Silicon via MLX:

- **Falcon Perception** (tiiuae/Falcon-Perception) — 3B-param VLM for expression-based segmentation, detection, and OCR
- **SAM 3.1** (mlx-community/sam3.1-bf16) — Meta's Segment Anything Model 3.1, MLX-community BF16 variant for video tracking and multi-prompt segmentation
- **Gemma 4 26B** (mlx-community/gemma-4-26b-a4b-it-4bit) — Google's 26B MoE (~3.8B active params), 4-bit quantized, as the reasoning/reasoning layer

**Design principle:** zero modifications to any existing project. VisionBrain reads from cached weights and the Falcon-Perception git repo, imports from them, and never writes back.

---

## Pipeline Architecture

```
Drone footage (MP4)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ SAM 3.1 — Frame-by-frame tracking + masks                   │
│ Prompts: "cow", "sheep", "fence post", "crop row"          │
│ Output: per-frame detections + object IDs                 │
└──────────────────────────────────────────────────────────────┘
    │
    ▼ (key frames only)
┌──────────────────────────────────────────────────────────────┐
│ Falcon Perception — Refined expression-based segmentation    │
│ Prompts: "lame cattle", "yellowed crop rows"                  │
│ Output: pixel-accurate masks + attribute detection          │
└──────────────────────────────────────────────────────────────┘
    │
    ▼ (structured mask metadata)
┌──────────────────────────────────────────────────────────────┐
│ Gemma 4 26B — Reasoning + field report generation             │
│ Input: structured detections + natural-language questions   │
│ Output: field reports, anomaly detection, behavioral analysis│
└──────────────────────────────────────────────────────────────┘
```

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
│       ├── gemma_inference.py ← Gemma 4: ask(), generate_report(), gemma_available()
│       ├── viz.py            ← Set-of-Marks rendering, crop extraction, relations
│       ├── agent_tools.py    ← agent-facing: ground_expression(), compute_relations()
│       ├── agent_loop.py     ← VLM agent: tool loop, context pruning
│       └── cli.py            ← CLI commands
├── tests/
│   └── test_visionbrain.py
└── assets/
    └── samples/              ← test images and output
```

---

## Module Specifications

### `loader.py` — Model Registry

**Public API:**
- `falcon_perception_record() -> ModelRecord`
- `sam31_record() -> ModelRecord`
- `all_records() -> list[ModelRecord]`
- `print_status()`
- `falcon_repo() -> Path`
- `sam31_cache_path() -> Path | None`

**Model variants:**
- SAM 3.1 uses `mlx-community/sam3.1-bf16` — public MLX-community conversion, no gated access needed
- Gemma 4 26B uses `mlx-community/gemma-4-26b-a4b-it-4bit` — public, no gated access needed

---

### `fp_inference.py` — Falcon Perception Pipeline

**Public API:**
- `segment(image, expression, *, ...) -> (list[MaskResult], InferenceStats)`
- `detect(image, expression, *, ...) -> (list[DetectionResult], InferenceStats)`
- `ocr(image, question, *, ...) -> (list[DetectionResult], str, InferenceStats)`

**MaskResult fields:** `mask_id`, `centroid_x/y`, `bbox_x1/y1/x2/y2`, `area_fraction`, `image_region`, `rle`

**DetectionResult fields:** `label`, `score`, `cx`, `cy`, `h`, `w`

**InferenceStats fields:** `preprocess_ms`, `generation_ms`, `total_ms`, `prefill_tokens`, `decoded_tokens`, `tokens_per_sec`, `n_masks`, `n_detections`

---

### `sam3_inference.py` — SAM 3.1 Wrapper

**Public API:**
- `sam31_available() -> bool`
- `detect_multi(image, prompts, *, threshold, resolution, task) -> list[Sam31Detection]`
- `track_video(video_path, prompts, output_path, *, threshold, every_n_frames, backbone_every, resolution, opacity) -> VideoTrackStats`
- `track_realtime(camera_or_video, prompts, *, ...) -> None`

**Sam31Detection fields:** `label`, `score`, `bbox_xyxy`, `mask` (optional np.ndarray)

**VideoTrackStats fields:** `total_frames`, `processed_frames`, `fps`, `unique_objects`, `output_path`

**Weight download:** `huggingface-cli download mlx-community/sam3.1-bf16` (public, no auth required)

---

### `gemma_inference.py` — Gemma 4 26B Reasoning Layer

**Public API:**
- `gemma_available() -> bool`
- `ask(question, *, detections, frame_history, image_path, max_tokens, temperature, kv_bits, kv_quant_scheme) -> GemmaResponse`
- `generate_report(summary_text, *, report_type, max_tokens, temperature, kv_bits, kv_quant_scheme) -> GemmaResponse`
- `unload_gemma() -> None` — free ~15.6GB RAM

**GemmaResponse fields:** `text` (str), `stats` (GemmaStats)

**GemmaStats fields:** `prompt_tokens`, `generation_tokens`, `prompt_tps`, `generation_tps`, `decode_ms`

**KV cache optimization:** `--kv-bits 3.5 --kv-quant-scheme turboquant` (30-64% memory savings, up to 1.16x speedup at longer contexts)

**Weight download:** `huggingface-cli download mlx-community/gemma-4-26b-a4b-it-4bit` (~15.6GB, public)

---

### `viz.py` — Visualization

**Public API:**
- `render_som(image, masks, *, ...) -> PIL.Image`
- `render_detections(image, detections, *, ...) -> PIL.Image`
- `get_crop(image, mask, *, pad=0.05) -> PIL.Image`
- `compute_relations(masks) -> dict`

---

### `agent_tools.py` — Agent Tools

**Public API:**
- `run_ground_expression(image, expression, *, ...) -> dict[int, dict]`
- `compute_relations(masks, mask_ids) -> dict`
- `masks_to_vlm_json(masks) -> list[dict]`

---

### `agent_loop.py` — VLM Agent

**Public API:**
- `VLMClient(api_key, model, base_url)`
- `run_agent(image, question, client, *, ...) -> AgentResult`

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
| `visionbrain analyze` | Full pipeline: SAM 3.1 track → Gemma 4 reasoning → report |
| `visionbrain agent` | VLM-powered visual reasoning |

#### `analyze` command

```bash
visionbrain analyze --video drone.mp4 --query "cattle in the pasture" --report

# Options
--video             Input video (required)
--query             Natural-language query (required)
--prompts           SAM 3.1 text prompts (default: use --query)
--output            Output annotated video path
--json-output       Per-frame detection JSON path
--report-output     Field report text path
--threshold         Detection confidence (default 0.15)
--every             Run SAM detection every N frames (default 2)
--backbone-every    Re-run ViT backbone every N detections (default 1)
--resolution        SAM input resolution (default 1008)
--opacity           Mask overlay opacity (default 0.6)
--sample-frames     Frames to sample for Gemma reasoning (default 10)
--report            Generate written field report via Gemma 4
--report-type       field | brief | json (default: field)
--question          Custom question for Gemma 4
--max-tokens        Max output tokens (default 512)
```

---

## One-Time Setup

```bash
# SAM 3.1 weights — MLX community variant, public (no auth needed)
huggingface-cli download mlx-community/sam3.1-bf16

# Gemma 4 26B weights — public (no auth needed)
huggingface-cli download mlx-community/gemma-4-26b-a4b-it-4bit
```

---

## Dependencies

Core:
- `mlx`
- `mlx_vlm`
- `transformers`
- `pillow`
- `pycocotools`
- `numpy`
- `opencv-python` (SAM video tracking)

Run: `FALCON_PY=~/Library/Caches/pypoetry/virtualenvs/falcon-perception-NVnkjaN--py3.12/bin/python`
`$FALCON_PY -m pytest tests/ -v`
