# VisionBrain — Technical Specification

> Aerial & camera vision AI on Apple Silicon. SAM 3.1 runs locally; Gemma 4 e2b runs via Ollama on localhost.

---

## Architecture

**Three-model backend design** — Gemma 4 is auto-selected from custom → Ollama → remote → local MLX based on availability (custom = any user-configured OpenAI-compatible VLM endpoint).

```
THIS MAC MINI (100.72.41.118, Mac Mini M4 16GB)
  SAM 3.1 (mlx-community/sam3.1-bf16) — local MLX
  └── Per-frame detection JSON (track IDs, centroids, bboxes, area fractions)
  └── Annotated stills (always produced)
  └── Annotated MP4 (opt-in, --include-video)
           │
           │  Structured detection JSON + semantic question
           ▼
  GEMMA BACKEND (auto-selected by available_backend()):
    Custom (user-configured OpenAI-compatible endpoint) — ~/.visionbrain/settings.json
    OR Ollama (localhost:11434) — gemma4:e2b, 7.2GB — preferred
    OR Remote (http://100.72.41.118:8080) — mlx-community/gemma-4-26b-a4b-it-4bit
    OR Local MLX — mlx-community/gemma-4-26b-a4b-it-4bit, ~32GB RAM
  └── Field reports, Q&A, anomaly detection
```

---

## Overview

VisionBrain is a Python library and CLI providing operator-friendly access to three state-of-the-art vision models, running entirely locally on Apple Silicon:

- **Falcon Perception** (tiiuae/Falcon-Perception) — 3B-param VLM for expression-based segmentation, detection, and OCR
- **SAM 3.1** (mlx-community/sam3.1-bf16) — Meta's Segment Anything Model 3.1, MLX-community BF16 variant for video tracking and multi-prompt segmentation
- **Gemma 4 e2b** (gemma4:e2b via Ollama) — Google's 2B MoE (~0.6B active params), 8-bit quantized, via Ollama on localhost:11434 — the reasoning/report layer

**Design principle:** zero modifications to any existing project. VisionBrain reads from cached weights and the Falcon-Perception git repo, imports from them, and never writes back.

---

## Pipeline Architecture

```
Drone footage (MP4)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Prompt Router — splits user query into SAM targets +          │
│                semantic reasoning question                    │
└──────────────────────────────────────────────────────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌────────────────┐ ┌──────────────────────────────────────────────┐
│ SAM 3.1        │ │ Falcon Perception + Gemma 4                  │
│ Concrete       │ │ Semantic reasoning + field reports          │
│ segmentable    │ │                                            │
│ objects        │ │ Input: semantic query + detections from SAM │
│ (cow, fence,   │ │ Output: reasoning, anomaly detection,        │
│  roof, etc.)   │ │        behavioral analysis, field reports    │
└────────────────┘ └──────────────────────────────────────────────┘
    │
    ▼ (per-frame detection JSON — primary output)
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
│       ├── sam3_inference.py ← SAM 3.1: detect_multi(), track_video(), track_video_with_json()
│       ├── frame_selector.py  ← Fast Falcon scorer: score_frames()
│       ├── gemma_inference.py ← Gemma 4: ask(), generate_report(), gemma_available(), available_backend()
│       ├── prompt_router.py   ← Query routing: SAM targets + semantic question
│       ├── viz.py            ← Set-of-Marks rendering, crop extraction, relations
│       ├── agent_tools.py    ← agent-facing: ground_expression(), compute_relations()
│       ├── agent_loop.py     ← VLM agent: tool loop, context pruning
│       ├── cli.py            ← CLI commands
│       └── web_app.py        ← FastAPI ground control (port 7860)
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
- `falcon_ocr_record() -> ModelRecord` — registry-only entry for `tiiuae/Falcon-OCR` (0.3B OCR companion: text, tables, formulas); `can_load` is always False — upstream serving is vLLM/CUDA, no MLX inference path in VisionBrain yet
- `all_records() -> list[ModelRecord]` — 4 entries: Falcon Perception, SAM 3.1, Ollama Gemma, Falcon-OCR
- `print_status()`
- `falcon_repo() -> Path`
- `sam31_cache_path() -> Path | None`

**Model variants:**
- SAM 3.1 uses `mlx-community/sam3.1-bf16` — public MLX-community conversion, no gated access needed
- Gemma 4 e2b uses `gemma4:e2b` via Ollama — 7.2 GB, managed by Ollama (no HuggingFace cache needed)
- Falcon-OCR uses `tiiuae/Falcon-OCR` — OCR companion (text, tables, formulas); registry-only, served upstream via vLLM/CUDA

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
- `track_video_with_json(video_path, prompts, output_path, json_path, *, threshold, every_n_frames, backbone_every, resolution, opacity, contour_thickness, adaptive_motion, motion_threshold, propagate_frames, relevance_scores, relevance_threshold) -> tuple[VideoTrackStats, list[dict]]`
- `track_realtime(camera_or_video, prompts, *, ...) -> None`

**Adaptive Parameters** (all disabled by default):
- `adaptive_motion`: enable motion-guided frame skipping using greyscale pixel delta
- `motion_threshold`: frame delta threshold (lower = more sensitive, default 0.03)
- `propagate_frames`: reuse last detection masks for N frames after each detect
- `relevance_scores`: `{frame_index: relevance}` dict from Falcon fast-scan
- `relevance_threshold`: minimum relevance to process a frame (default 0.2)

**Weight download:** `huggingface-cli download mlx-community/sam3.1-bf16` (public, no auth required)

### `frame_selector.py` — Fast Falcon Frame Scorer

**Public API:**
- `score_frames(video_path, query, *, sample_every_n_seconds, max_frames, resolution, min_relevance) -> FrameScores`

(The `visionbrain fastscan` CLI wrapper is `cmd_fastscan()` in `cli.py`, not in this module.)

**FrameScores fields:** `video_path`, `total_frames`, `fps`, `duration_s`, `frames_scored`, `is_relevant`, `quick_answer`, `regions`, `frame_scores`

**TemporalRegion fields:** `start_time`, `end_time`, `avg_relevance`, `label`

**Algorithm:** Extract frames at uniform intervals → Falcon detect at low-res → relevance scoring → temporal region clustering → natural-language quick answer.

**Weight download:** `huggingface-cli download mlx-community/sam3.1-bf16` (public, no auth required)

### `gemma_inference.py` — Gemma 4 Reasoning Layer (Consolidated)

**Backends:** Custom → Ollama → Remote server → Local MLX (auto-selected by availability)

**Custom backend settings store:** `~/.visionbrain/settings.json` (schema `{"base_url": str, "model": str, "api_key": str}`, written with 0600 permissions) — lets ask/report use any OpenAI-compatible endpoint (LM Studio, vLLM, OpenRouter, OpenAI) via `POST {base_url}/chat/completions`. `api_key` is never included in API responses or `save_vlm_settings()`'s return value.

**Public API:**
- `available_backend() -> str | None` — 'custom' | 'ollama' | 'remote' | 'local' | None
- `settings_path() -> Path` — settings file location (`~/.visionbrain/settings.json`)
- `load_vlm_settings(path=None) -> dict` — `{"base_url", "model", "api_key"}`; missing/corrupt file → all empty strings; never raises
- `save_vlm_settings(base_url="", model="", api_key="", clear_key=False, path=None) -> dict` — overwrites only provided non-empty values; `clear_key=True` wipes only the key; best-effort write (never raises); returns the stored settings with `api_key` redacted
- `custom_backend_configured() -> bool` — True when both base_url and model are set
- `gemma_available() -> bool` — True if any backend is available
- `ask(question, *, detections, frame_history, image_path, max_tokens, temperature, kv_bits, kv_quant_scheme) -> GemmaResponse`
- `generate_report(summary_text, *, report_type, max_tokens, temperature, kv_bits, kv_quant_scheme) -> GemmaResponse`
- `unload_gemma() -> None` — releases local MLX weights from cache
- `test_connection() -> dict` — smoke test the active backend (the custom branch reports the saved settings summary without a network probe)

**GemmaResponse fields:** `text` (str), `stats` (GemmaStats)

**GemmaStats fields:** `prompt_tokens`, `generation_tokens`, `prompt_tps`, `generation_tps`, `decode_ms`

**Backend priority:**
1. **Custom** — user-configured OpenAI-compatible endpoint; `Authorization: Bearer` sent only when an api_key is saved
2. **Ollama** (`gemma4:e2b`, 7.2GB) — localhost:11434, preferred for local Mac
3. **Remote** (`mlx-community/gemma-4-26b-a4b-it-4bit`) — http://100.72.41.118:8080
4. **Local MLX** (`gemma-4-26b-a4b-it-4bit`) — requires ~32GB RAM

**Note:** Ollama gemma4:e2b requires `max_tokens >= 200` for structured reasoning.

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
| `visionbrain analyze` | Full pipeline: SAM 3.1 track → Falcon key-frames (optional) → Gemma 4 reasoning → report |
| `visionbrain agent` | VLM-powered visual reasoning |
| `visionbrain fastscan` | Fast Falcon-only scan: relevance answer in seconds |

#### `analyze` command

**Output prioritization:** JSON timeline and annotated stills are the primary outputs (always produced). Annotated MP4 video is opt-in via `--include-video` (disabled by default).

```bash
visionbrain analyze --video drone.mp4 --query "vehicles blocking the north access road" --report

# With annotated video (opt-in)
visionbrain analyze --video drone.mp4 --query "person" --include-video --report

# Fast-path: get quick answer in <60s, then continue full analysis
visionbrain analyze --video drone.mp4 --query "person" --fast --report

# Adaptive: motion skip + mask propagation for long videos
visionbrain analyze --video drone.mp4 --query "person" --adaptive --propagate 5 --every 8
```

```bash
# Options
--video             Input video (required)
--query             Natural-language query (required)
--prompts           SAM 3.1 text prompts (default: use --query, parsed by prompt_router)
--include-video     Generate annotated MP4 video (disabled by default)
--output            Output video path (requires --include-video)
--json-output       Per-frame detection JSON path (always produced)
--report-output     Field report text path
--review-reel-output Keyframe review reel MP4 path
--hold-seconds      Seconds to hold each analyzed frame in review reel (default 2.0)
--still-dir         Annotated stills directory (always produced, auto-generated by default)
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
--falcon-refine     Run Falcon Perception on K key frames for semantic deep-dive
--falcon-frames     Number of key frames to pass to Falcon (default 6)
# Fast-path + adaptive options
--fast              Run fast-path Falcon scan first, return quick answer immediately
--fast-output       Write fast-scan JSON result to this file
--adaptive          Enable adaptive SAM: motion-guided skip + relevance filter
--motion-threshold  Frame delta threshold for motion skip (default 0.03)
--propagate         Propagate masks forward N frames after each detect (default 0=off)
--relevance-filter  Skip frames where Falcon fast-scan scored relevance below threshold
--parallel-falcon   Process Falcon key-frames in parallel (default: True)
--sequential-falcon  Disable parallel Falcon processing
```

#### `track` command

```bash
visionbrain track --video drone.mp4 --prompts person car --output tracked.mp4

# Options
--video             Input video (required)
--prompts           SAM 3.1 text prompts to track (required)
--output            Output video path
--threshold         Detection confidence (default 0.15)
--every             Run detection every N frames (default 2)
--backbone-every    Re-run ViT every N detections (default 1)
--resolution        SAM input resolution (default 1008)
--opacity           Mask overlay opacity (default 0.6)
--json-output       Also write per-frame detections JSON at this path (uses track_video_with_json)
--supervision       Render with supervision annotators (mask/box/label)
--persistent-ids    ByteTrack persistent tracker IDs across occlusions
--adaptive-motion   Skip detection on low-motion frames
--motion-threshold  Grey-delta threshold for adaptive motion skip (default 0.03)
--propagate         Propagate last detection forward N frames after each detect (default 0)
```

### `detection_core.py` — Shared Detection Primitives

**Public API:**
- `box_iou(a, b) -> float` — IoU of two xyxy boxes in any shared coordinate space
- `labels_compatible(a, b) -> bool` — casefold/substring label match ("cow" vs "brown cow")
- `dedup_detections(items, *, threshold) -> list` — center-distance duplicate suppression (unit-agnostic; bridge passes normalized boxes + TII's 0.01)
- `PersistentTrackManager` — dependency-free session-local identity assignment (`assign(items, frame_id=, now_ms=)` → items with `track_id`/`color_id`/`track_state`/`track_confidence`; TTL + frame-gap expiry; ambiguity → "uncertain")
- `merge_validate(sam_items, falcon_items, mode="soft|hard|off") -> (overlay, llm_items, stats)` — cross-engine agreement merge
- Constants: `AGREE_IOU_THRESHOLD`, `VALIDATE_MODES`, `DEFAULT_TRACK_TTL_MS`, …

Pure Python, no MLX — canonical for web app, CLI, and the live bridge hub.

### `live_tracking.py` — Stateful Live SAM 3.1 Tracker

**Public API:**
- `LiveSamTracker(*, model, resolution, threshold, detect_every, backbone_every, tracker=None, backbone_fn=None, detect_fn=None, preprocess_fn=None, mask_to_polygon=None)`
- `.step(image, prompts, task, width, height, frame_id, timestamp_ms) -> list[dict]` — one frame in, normalized items out (`label/score/box/source="sam"/track_id/color_id/track_state/stale_ms`, optional `polygon`)
- `.reset()` — new shot/scene

ViT backbone cached across frames (recompute every `backbone_every`); between detects the last items are re-emitted held with `track_state="predicted"` and honest `stale_ms`. Loader and all three compute hooks are injectable → unit-testable without mlx_vlm or weights.

### `model_host.py` — Refcounted MLX Checkpoint Residency

- `HOST.acquire(key, loader) / HOST.release(key) / HOST.resident()` — payload-agnostic refcount cache; entry freed + `mx.clear_cache()` when the last holder releases. Lets engine and VLM share one LFM checkpoint.

### `mlx_compat.py` — mlx_vlm Load Compatibility Shims

- `apply_all()` — idempotent, thread-safe shims called before any `mlx_vlm.utils.load` (wired into `vlm_registry._load_checkpoint`, `gemma_inference._ensure_local_gemma`, and the bridge's `lfm._load_checkpoint`):
  - gemma 4: mlx_vlm 0.4.4's `ScaledLinear` gets a `to_quantized()` (→ `QuantizedScaledLinear` re-applying the scalar) so the quantized `per_layer_model_projection` loads; `Attention` stops allocating dead `k_norm/k_proj/v_proj` for KV-shared layers (checkpoints omit them).
  - LFM2.5-VL: wraps `load_config` to force `projector_use_layernorm=true` only when the checkpoint's weight index actually ships `multi_modal_projector.layer_norm` (checkpoint configs claim false).
  - No-ops permanently once mlx_vlm ships equivalent support. NOTE: LFM loads additionally require torch+torchvision for the image processor; without them the LFM engine cannot load regardless.

### `vlm_registry.py` — Hot-Swappable Local VLMs (ask/report)

- `MODELS = {"gemma": gemma-4-e2b-it-4bit, "lfm": LFM2.5-VL-450M-MLX-4bit, "lfm3b": LFM2.5-VL-3B-MLX-4bit}`
- `set_model(key) / current_key() / current_model() / available()`
- `ask(question, detections=None, prompts=None, image=None) -> str`; `generate_report(summary_text, report_type="field", image=None) -> str` — multimodal generation via mlx_vlm with shared sampling policy (min_p=0.15, repetition_penalty=1.05)
- `position_label(x, y)`, `format_detection_lines(dets)` — shared plain-language rendering for prompts/HUDs
- `SYSTEM_PROMPT` — anti-hallucination copilot contract (detector counts are authoritative but never a full inventory)

### `prompt_router.py` — Query Routing

**Public API:**
- `route(query: str) -> PromptResult` — splits a user query into open-vocabulary SAM targets and the semantic question
- `route_fallback(query: str) -> list[str]` — returns default SAM prompts if route() produces no targets
- `PromptResult` dataclass: `segment_targets: list[str]`, `semantic_query: str`, `original_query: str`, `routed_from: str`
- `STOPWORDS: frozenset[str]` — generic closed-class words only (articles, conjunctions, prepositions, auxiliaries, filler verbs); the module contains no domain vocabulary
- `MAX_TARGETS: int = 8` — cap on SAM prompts per query (multiplex sanity limit)

**Routing logic (open-vocabulary pass-through):**
- SAM 3.1 is open-vocab: the token stream is partitioned into noun phrases at stopwords, and every phrase is passed through as typed — no whitelist, no stemming (plurals and multi-word phrases like "yellow school bus" work natively)
- Phrases are deduped case-insensitively (first-seen order) and capped at `MAX_TARGETS` (8); empty phrases and pure numbers are dropped
- `semantic_query` is the full original query — Gemma reasons over the complete ask (no word-stripping)
- Empty or stopword-only queries → empty `segment_targets` (caller falls back via `route_fallback`); `routed_from` records how the split happened

**Usage:** `cmd_analyze` calls `route(args.query)` and passes `segment_targets` to SAM, `semantic_query` to Falcon/Gemma.

#### `fastscan` command

```bash
# Fast Falcon-only scan: sub-60s relevance answer
visionbrain fastscan --video drone.mp4 --query "person"

# Options
--video             Input video (required)
--query             Natural-language query (required)
--every             Sample one frame every N seconds (default 5)
--max-frames       Maximum frames to score (default 60)
--resolution        Falcon resolution (default 360 — low-res for speed)
--min-relevance     Minimum relevance to count as a region (default 0.2)
--output            Write structured JSON result to this path
```

### `web_app.py` — Ground Control UI

FastAPI app serving the single-page Ground Control dashboard (`static/index.html`) on port 7860. Launch with `visionbrain ui`.

**API surface:**
- `GET /api/status` — model registry + cache status, `gemma_remote` availability flag, and `vlm` `{backend, custom_configured}` (one blocking backend probe feeds all three); `GET /api/healthz` — Gemma backend health
- `GET /api/settings` — saved custom VLM endpoint as `{configured, base_url, model, has_key}` (the api_key itself is never returned); `POST /api/settings` — JSON body `{base_url?, model?, api_key?, clear_key?}` → same shape as GET (bad JSON → 400)
- `POST /api/upload` — upload media, returns `file_id`
- `POST /api/job/{kind}` — start a job (`analyze`, `fastscan`, `detect`, `segment`, `ocr`, `track`, `sam3`, `agent`); each spawns the CLI as a subprocess and returns `{job_id}`. `agent` accepts optional `question`/`api_key`/`model`/`base_url` form fields (empty fields fall back to the saved VLM settings); `track` accepts optional `json_output`/`supervision`/`persistent_ids`/`adaptive_motion`/`motion_threshold`/`propagate`; `analyze` accepts optional `question` (forwarded to Gemma)
- `GET /api/job/{jid}` — job state + streamed output; `GET /api/job/{jid}/stream` — SSE stream (phase, heartbeat, progress)
- `GET /api/job/{jid}/detections|report|fast|file/{kind}` — result artifacts

**UI layout:**
- Header: logo, mode tabs (analyze / detect / segment / track / sam-3 / ocr), connection status
- Left rail: MODEL INTEL, Ollama server status, LAST MISSION stats, quick actions/downloads
- Center: media drop zone → mission progress → annotated video/image results
- Right rail (top-to-bottom): **MISSION SETUP** (all form controls for the active tab, vertical, scrollable) → OPERATIONS LOG → result panels (detection summary, fastscan quick answer, field report)

Controls live in the right-rail MISSION SETUP panel — there is deliberately no bottom config bar.

---

## One-Time Setup

```bash
# SAM 3.1 weights — MLX community variant, public (no auth needed)
huggingface-cli download mlx-community/sam3.1-bf16

# Gemma 4 e2b via Ollama — no HuggingFace download needed
# Just ensure Ollama is running and gemma4:e2b is available:
ollama list | grep gemma4
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
