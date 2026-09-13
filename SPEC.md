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
│       ├── pilot_eval.py     ← Pilot measurement: run_pilot_eval(), PilotReport
│       ├── gemma_inference.py ← Gemma 4: ask(), generate_report(), gemma_available(), available_backend()
│       ├── prompt_router.py   ← Query routing: SAM targets + semantic question
│       ├── viz.py            ← Set-of-Marks rendering, crop extraction, relations
│       ├── agent_tools.py    ← agent-facing: ground_expression(), compute_relations()
│       ├── agent_loop.py     ← VLM agent: tool loop, context pruning
│       ├── cli.py            ← CLI commands
│       └── web_app.py        ← FastAPI ground control (port 7860)
├── tests/                    ← pytest suites (test_visionbrain.py plus per-module files)
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

**FrameScores fields:** `video_path`, `total_frames`, `fps`, `duration_s`, `frames_scored`, `is_relevant`, `quick_answer`, `regions`, `frame_scores`, `frames_failed` (inference failures, kept as `failed=True` FrameScores), `sampled_span_s` (first→last sampled timestamp)

**FrameScore fields:** `frame_index`, `timestamp`, `relevance_score`, `detection_count`, `top_label`, `has_query_match`, `failed`

**TemporalRegion fields:** `start_time`, `end_time`, `avg_relevance`, `label`

**Algorithm:** Extract frames at uniform intervals → Falcon detect at low-res → relevance scoring → temporal region clustering → natural-language quick answer. Sampling honesty: when candidates exceed `max_frames`, `_select_sample_indices()` picks an evenly spaced subset spanning the FULL video (never just the earliest frames); per-frame Falcon exceptions are recorded as `failed=True` frames and counted in `frames_failed`; a negative quick answer states the sampled coverage span, frames scored, and failed count rather than claiming absence over the whole video.

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
- `VLMClient(api_key, model, base_url)` — `chat(messages, tools=None) -> ChatResponse` (native `tool_calls` are mapped into `ChatResponse.tool_calls` as `{"name", "parameters"}`; malformed JSON arguments fall back to `parameters={}` plus `arguments_error`/`arguments_raw` markers)
- `ChatResponse` — `content: str`, `tool_calls: list[dict]`
- `run_agent(image, question, client, *, ...) -> AgentResult`

**Tool-call contract:** `run_agent` accepts a `ChatResponse` or a legacy plain-`str` return (subclassed clients keep working) via `_normalize_chat_response`. Native `tool_calls[0]` takes precedence; the textual `<tool>...</tool>` protocol is the fallback. Neither present → the loop raises a clear `ValueError`. Malformed native arguments are surfaced to the VLM as a retryable error message, never dispatched with empty parameters.

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
| `visionbrain pilot-eval` | Replay footage + ground-truth labels → honest metrics report (missed events, false alerts, latency, coverage) |

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
- `mask_to_polygon(mask, width, height, max_points=48) -> list | None` — dependency-free mask outline tracer: per-row pixel runs traced top-down then back up, normalized 0-1 (numpy imported lazily; `None` on empty/invalid masks or missing numpy). Feeds the `polygon` field on live/tracking items so clients paint the object's mask shape instead of a box around it
- `PersistentTrackManager` — dependency-free session-local identity assignment (`assign(items, frame_id=, now_ms=)` → items with `track_id`/`color_id`/`track_state`/`track_confidence`; TTL + frame-gap expiry; ambiguity → "uncertain")
- `merge_validate(sam_items, falcon_items, mode="soft|hard|off") -> (overlay, llm_items, stats)` — cross-engine agreement merge
- Constants: `AGREE_IOU_THRESHOLD`, `VALIDATE_MODES`, `DEFAULT_TRACK_TTL_MS`, …

Pure Python, no MLX — canonical for web app, CLI, and the live bridge hub.

### `crosscheck.py` — SAM-vs-Falcon Cross-Engine Validation

Pure (stdlib + `detection_core.box_iou` only):
- `crosscheck(sam_dets, falcon_dets, *, iou_threshold=0.5) -> CrosscheckResult` — greedy highest-IoU-first one-to-one matching between two detection lists (`{"bbox_xyxy", "label", "score"?}`); a pair matches when `iou >= threshold` (labels are reported per pair but not required to agree, so disagreement stays visible). `CrosscheckResult`: `frame_index, matched, sam_only, falcon_only, agreement, matches` — agreement = `matched / max(1, max(len(sam), len(falcon)))`
- `falcon_to_dets(detection_results, orig_w, orig_h)` — Falcon normalized `cx/cy/h/w` → pixel `bbox_xyxy` dicts
- `summarize(results) -> dict` — aggregate `{frames, matched, sam_only, falcon_only, agreement}` (mean per-frame agreement; 0.0 when empty)

`cmd_analyze` runs it automatically on Falcon-refined key frames (unless `--no-crosscheck`), prints a per-frame + aggregate block to the ops log (failures warn and never break the pipeline), and feeds one compact agreement line into Gemma's reasoning context.

### `grounding.py` — LFM Grounding Third Opinion

Pure (stdlib only; module imports with no MLX/PIL):
- `build_grounding_prompt(targets)` — demands one line per instance: `<box>x1,y1,x2,y2</box> label` with integer 0–1000 coordinates, exactly `NONE` when nothing found
- `parse_grounding_boxes(text, width, height) -> list[dict]` — tolerant parser (canonical `<box>` tags, parenthesized `(a,b),(c,d)`, JSON arrays) with a per-box scale heuristic (≤1.5 → 0-1, ≤100 → 0-100, else 0-1000); clamps to image bounds, orders corners, `[]` on NONE
- `grounding_crosscheck(sam_dets, boxes, *, iou_threshold=0.3)` — `crosscheck.crosscheck` wrapper at the looser VLM-appropriate threshold (VLM boxes are coarse)
- `python -m visionbrain.grounding <image> <target>…` — probe block printing the raw model reply next to the parsed boxes (human verifies format)

`cmd_analyze --lfm-ground` (opt-in, default off) asks the local LFM VLM to ground the SAM targets on the Falcon key frames, cross-checks against SAM, prints the ops-log block, and appends one agreement line to Gemma's context; any failure degrades to a single warning line.

### `live_tracking.py` — Stateful Live SAM 3.1 Tracker

**Public API:**
- `LiveSamTracker(*, model, resolution, threshold, detect_every, backbone_every, tracker=None, backbone_fn=None, detect_fn=None, preprocess_fn=None, mask_to_polygon=None)` — `mask_to_polygon` defaults to `detection_core.mask_to_polygon` (pass `None`-producing custom or `lambda *a: None` to disable)
- `.step(image, prompts, task, width, height, frame_id, timestamp_ms) -> list[dict]` — one frame in, normalized items out (`label/score/box/source="sam"/track_id/color_id/track_state/stale_ms`, plus `polygon` whenever the task yields masks)
- `.reset()` — new shot/scene

ViT backbone cached across frames (recompute every `backbone_every`); between detects the last items are re-emitted held with `track_state="predicted"` and honest `stale_ms`. Loader and all three compute hooks are injectable → unit-testable without mlx_vlm or weights.

### `live_engine.py` — Local Live Engine (field-hub WS + smart capture)

Lets VisionBrain itself play the field-hub server role: one worker streams SAM 3.1 over `WS /api/live/ws` in the exact hub binary format (`>III` header + JPEG + `>I` + telemetry JSON), so the unmodified browser client renders it. `configure(uploads_dir, clips_dir=None)` pins the upload resolution dir and the clip output dir (created when missing; `None` disables capture). On every detect frame the worker extracts each SAM mask's outline via `detection_core.mask_to_polygon` and emits it on the item as `polygon` (normalized `[[x, y], ...]`, optional); the dashboard paints filled mask shapes and falls back to the box for items without one. Held re-publishes carry the polygon through unchanged.

**Controls** (inbound JSON; key `"type"`, `"action"` accepted as alias): `start` (file/webcam/url + optional `threshold`/`detect_every`/`resolution`/`backbone_every` (DEPRECATED — still validated for old clients, then stripped before the worker sees it; a status note answers "backbone_every is deprecated and ignored — features are recomputed on every detection pass")/`jpeg_quality` (30-95, default 70)/`send_width` (256-3840, default 1280)/`task` (`"segment"`|`"detect"`, default `"segment"`)), `set_prompts` (an EMPTY list is valid and means "detection off" — the hub-protocol pause semantics; the worker skips the model and emits empty detection sets until prompts return, and the pause CLEARS the stored ask/report evidence so earlier counts cannot survive as current observations), `set_threshold` (live re-apply, no restart), `set_stream` (live `jpeg_quality`/`send_width` retune — the Android STREAM knobs), `set_task` (live MASK switch: `"segment"` emits polygons, `"detect"` skips polygon tracing for fast boxes), `set_engine` (hub-protocol chips; the local engine runs SAM only and answers with an honest status note), `set_vlm` (`gemma`|`lfm`|`lfm3b` via `vlm_registry.set_model`), `ask {question ≤ 500}` (replies `ask_ack` + `answer`, or `error` on rejection/failure; at dispatch the handler snapshots (full-res frame, detection records, active prompts) as ONE consistent server-owned evidence unit, then a background thread answers from that snapshot alone — it never re-reads live worker state; one in-flight slot shared with `report`), `report {summary?, report_type?}` (replies `report_result`, or `error` on rejection/failure; the counts line is derived SERVER-SIDE from the snapshot's detection records — the inbound `summary` is accepted for wire compatibility but is never used as detection evidence or forwarded to the model), `add_prompt_box` (draw a rectangle on the canvas → a persistent ROI "target N"; ≤8, pending before a worker like zones; NOTE: the installed mlx_vlm build plumbs the `boxes` kwarg but never applies box conditioning, so targets are pure ROI RELABELS of the single main detection pass — no per-target inference), `remove_targets`, `set_zones`, `set_triggers`, `set_watch`, `stop`, `shutdown`; `hello` → ignored. Anything invalid → `("unknown", {})` + status note, never an exception. `url` sources (`rtsp://`, `rtsps://`, `http://`, `https://` only) are gated by `validate_stream_url()` and always rendered via `redact_url()` (userinfo → `user:***@`) in status notes; a stream that fails to open (or 40 consecutive read failures) fails cleanly with `engine_stopped` — no auto-retry in the first cut. **Auth:** when `VB_TOKEN` is set, the WS requires `?token=` (checked before accept, close 4401) — HTTP middleware never sees WebSocket scopes, so the gate lives in the handler.
- `{"type":"set_zones","zones":[...]}` — REPLACES the set: `{"kind":"line","name"?,"a":[x,y],"b":[x,y]}` or `{"kind":"rect","name"?,"x1","y1","x2","y2"}`, normalized 0-1 (rect needs `x1<x2`, `y1<y2`), ≤ 40-char names (default `"zone N"`), max 8 — `validate_zones()`. Accepted before a worker exists (held pending, applied on start) and while running (rebuilt under the state lock on the next detect frame, which resets line counters). Ack `zones set (N)`.
- `{"type":"set_triggers","line_cross"?,"direction"?,"dwell_s"?,"clip"?,"pre_s"?,"post_s"?}` — partial merge over defaults (False / `"none"` / 0=off / True / 6 / 4); `direction` ∈ none|any|8-way compass. Ack `triggers set`.
- `{"type":"set_watch","enabled"?,"condition"?,"interval_s"? (1-30, default 4),"model"? ("lfm"|"lfm3b")}` — Ack `watch on`/`watch off`.

**New outbound JSON:** `{"type":"error","error":"…"}` (ask/report path ONLY — no engine running, no current observation, prompts paused, the shared slot busy, or backend VLM inference failed; every other failure path keeps its `status` notes, and no client should parse status text to detect an ask/report failure; the grounding rejections (no engine / no observation / paused) are decided BEFORE the shared inference slot is claimed — no VLM call — and an observed EMPTY set IS a valid observation), `{"type":"event","event":{kind: line_cross|direction|dwell|watch|zone_enter|zone_exit, zone, direction, track_id|null, ts, frame_id, detail}}` and `{"type":"capture","clip":{name, url:"/api/clips/<name>", kind}}`. Rect zones fire enter/exit per track transition (`RectZone`); line zones run `zones.LineZoneCounter` per detect frame on pixel boxes (totals increment → `line_cross`); `DirectionTriggerState` fires once per (track, heading) and re-arms on heading change (`"any"` = any real heading); `DwellTracker` fires once per stationary stretch after `dwell_s`.

**Clips:** the worker rings the same encoded JPEGs it streams (`maxlen = min(pre_s·fps or 30, 150)`); a fired trigger claims THE one capture slot — it spans post-roll collection, the encode queue, AND encoding (`request_capture` refuses while the capture exists in any state; later triggers still emit their events but never allocate a second capture) — snapshots pre-roll, accumulates until `trigger_ts + post_s`, and hands the capture to a dedicated writer daemon thread (decode + `mp4v` re-encode must never stall the read/detect/encode loop) which starts LAZILY on the first handoff (`_ensure_capture_writer` — a worker that never captures starts no thread), writes `clips_dir/clip_<ts>_<kind>.mp4`, announces it, and prunes the dir to the 50 newest files. The writer releases the slot in its `finally` — success AND failure. Every worker exit path funnels through `run()`'s `finally` and sends the writer's `None` queue sentinel (`_shutdown_capture_writer`) AFTER any accepted capture, so FIFO ordering lets queued/encoding captures finish; an INCOMPLETE (still-collecting) post-roll capture is discarded; the worker joins the writer with an 8s timeout and the event loop never joins it. `web_app` serves `GET /api/clips/{name}` after `sanitize_clip_name()` (alnum/`_.-` + `.mp4` only; else 404).

**Fresh observations + streams:** EVERY scheduled detect pass recomputes image features from that pass's frame — the cross-pass backbone/encoder cache and `backbone_every` cadence are gone (`detect_every` is the sole inference throttle; within one pass the model helper may still cache as needed). Skipped passes and empty prompts do no inference; box targets add zero model calls; pause/resume and file looping can never reuse features from an earlier scene. Outbound replaceable streams are COALESCED latest-value-wins per viewer (one-slot buffer + sentinel each): binary frames AND `detections` sets — an EMPTY detection set is a real value that supersedes older objects (`None`, not `[]`, marks an empty slot), so a slow dashboard can never build a stale backlog of either. Events, captures, status, and ask/report results keep the in-order never-dropped queue — this bounds those two streams specifically, not every possible outbound source.

**Design:** single instance per process (module handle + `_engine_lock`) but MULTI-VIEWER: every connected socket gets a sink (in-order JSON queue + one-slot latest-frame AND latest-detections buffers) and the worker broadcasts to all of them, so several dashboards watch the same stream; a `start` while alive attaches the requester as a viewer (status note) instead of starting a second engine, and the engine stops when the last viewer disconnects. Worker + watcher are daemon threads. `vlm_registry` (and all heavy deps) import inside thread bodies, so CI imports the module and tests the pure helpers (`pack_frame`, `make_item`, `validate_control`, `validate_zones`, `sanitize_clip_name`, `validate_stream_url`, `redact_url`, `RectZone`, `DwellTracker`, `DirectionTriggerState`) with no MLX/weights. The watch thread sleeps `interval_s`, skips ticks when the worker is idle/busy, asks the local VLM `"…Answer with exactly YES or NO. Condition: …"` on the latest full-res frame, fires a `watch` event (+capture) on YES; errors → status notes throttled to 1/30s, and a model that never loads disables the watch with one note.

### `service.py` — Shared-Token Auth + Job-Slot Queue

Pure asyncio/stdlib primitives for LAN/business deployments (no fastapi or MLX imports — importable anywhere):
- `token_enabled() / check_token(provided)` — VB_TOKEN env (read at call time); constant-time compare via `hmac.compare_digest`; unset/empty disables auth entirely
- `max_jobs() -> int` — `VB_MAX_JOBS` clamped to 1..4 (default 1); invalid → default
- `JobQueue` — asyncio FIFO slot limiter: `acquire(key) -> position` (0 = started immediately, 1 = first in line…), `release(key)`, `queued_count`, `wait_position(key)` (live line spot: 0 when running, 1-based while waiting); deque-of-futures so there is no busy waiting, and a waiter cancelled while queued is skipped cleanly and never consumes a slot

`web_app.py` wiring: when `VB_TOKEN` is set, every `/api/*` path except `/api/healthz` — and `/uploads/{fid}` media — requires the token via the `X-Auth-Token` header or `?token=` query (401 JSON otherwise). WebSocket scopes never pass through HTTP middleware, so the live WebSocket enforces the same token in-handler via `?token=` (checked before accept, close 4401). The heavy subprocess endpoints (`analyze`, `fastscan`, `track`, `agent`) run through a shared `JobQueue`; light image jobs (`detect`, `segment`, `sam3`, `ocr`) bypass it. Job dicts carry `queue_position`/`queued`, launch responses gain `{queued, queue_position}` (submit-time), and job state + SSE heartbeats report the live line position while a job still waits. The job store is evicted to the newest 200 finished jobs; a job whose subprocess fails to launch lands in `error` state (never stuck `running`).

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

#### `pilot-eval` command

Replays a recorded video with a ground-truth event label file through the FastScan
scorer and produces an honest metrics report — the measurement gate for a pilot
workflow (BUSINESS_CAPABILITY_RECON.md). Ground truth JSON:
`{"query": "person", "events": [{"label": "person", "start_s": 12.0, "end_s": 20.0, "min_count": 1}]}`.

```bash
visionbrain pilot-eval --video drone.mp4 --ground-truth labels.json --report pilot_report.json --evidence-dir evidence/

# Options
--video             Recorded video to replay (required)
--ground-truth      Ground-truth event label JSON path (required)
--report            Write the report JSON to this path
--evidence-dir      Save the first supporting frame per detected event as JPEG
--tolerance-s       Matching tolerance around each event window (default 3.0)
--sample-every      Sample one frame every N seconds (default 5)
--max-frames        Maximum frames to score (default 60)
--min-relevance     Minimum relevance to count as a detection (default 0.2)
--resolution        Falcon resolution (default 360)
```

**Report (`PilotReport`):** `events_total/detected/missed`, `missed_events`, `false_alert_count`/`false_alerts` (frame-level), `per_event` (detection, `latency_s`, supporting frames, evidence path), `coverage` (duration vs sampled span, frames scored/failed), `runtime_s`, and always-present `caveats` (sampling interval vs short events; Falcon labels derive from the query so label agreement is not independent confirmation; failures and partial coverage when they occur). `summary()` renders raw numbers only. Tests inject a fake `score_fn`/`frame_reader`; no inference is needed to evaluate the harness itself.

### `web_app.py` — Ground Control UI

FastAPI app serving the single-page Ground Control dashboard (`static/index.html`) on port 7860. Launch with `visionbrain ui`.

**API surface:**
- `GET /api/status` — model registry + cache status, `gemma_remote` availability flag, and `vlm` `{backend, custom_configured}` (one blocking backend probe feeds all three); `GET /api/healthz` — Gemma backend health
- `GET /api/settings` — saved custom VLM endpoint as `{configured, base_url, model, has_key}` (the api_key itself is never returned); `POST /api/settings` — JSON body `{base_url?, model?, api_key?, clear_key?}` → same shape as GET (bad JSON → 400)
- `POST /api/upload` — upload media, returns `file_id`
- `POST /api/job/{kind}` — start a job (`analyze`, `fastscan`, `detect`, `segment`, `ocr`, `track`, `sam3`, `agent`); each spawns the CLI as a subprocess and returns `{job_id}`. `agent` accepts optional `question`/`api_key`/`model`/`base_url` form fields (empty fields fall back to the saved VLM settings); `track` accepts optional `json_output`/`supervision`/`persistent_ids`/`adaptive_motion`/`motion_threshold`/`propagate`; `analyze` accepts optional `question` (forwarded to Gemma)
- `GET /api/job/{jid}` — job state + streamed output; `GET /api/job/{jid}/stream` — SSE stream (phase, heartbeat, progress)
- `GET /api/job/{jid}/detections|report|fast|file/{kind}` — result artifacts
- `GET /api/clips/{name}` — serve smart-capture clips (name sanitized; traversal/bad extensions → 404)

**Auth + concurrency (service.py):** set `VB_TOKEN` to require the token (X-Auth-Token header or `?token=`) on all `/api/*` except `/api/healthz` — off by default; the live WebSocket is covered too (enforced in-handler, close 4401 before accept). `VB_MAX_JOBS` (1..4, default 1) caps concurrent heavy jobs via a FIFO queue with `queue_position`/`queued` visible in job state, launch responses, and SSE heartbeats (live line position while queued). See DEPLOY.md.

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
