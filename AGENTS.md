# Repository Guidelines

## Project Overview

VisionBrain is an aerial & camera vision AI toolkit running on Apple Silicon via MLX. It provides a Python library and CLI for three models — Falcon Perception (segmentation/detection/OCR), SAM 3.1 (video tracking), and Gemma 4 26B (reasoning/reports) — orchestrated in a two-machine pipeline where SAM runs locally and Gemma runs on a remote GPU server.

## Project Structure

```
VisionBrain/
├── src/visionbrain/          # All source code
│   ├── cli.py                # CLI entry point (argparse subcommands)
│   ├── __main__.py           # `python -m visionbrain` shim (delegates to cli.main)
│   ├── loader.py             # Model registry and Hugging Face cache status
│   ├── fp_inference.py       # Falcon Perception: segment(), detect(), ocr()
│   ├── sam3_inference.py     # SAM 3.1: detect_multi(), track_video(), track_realtime()
│   ├── gemma_inference.py    # Gemma 4: ask(), generate_report()
│   ├── prompt_router.py      # Routes a text query to Falcon prompts / SAM labels
│   ├── frame_selector.py     # Motion-based frame scoring (score_frames) behind the fastscan CLI
│   ├── zones.py              # Line/polygon zone counters, ZoneManager
│   ├── supervision_bridge.py # Converts Falcon/SAM results to supervision.Detections + ByteTrack
│   ├── detection_core.py     # Shared detection primitives: IoU matching, identity, cross-engine validation (pure Python)
│   ├── direction_tracking.py # Per-track 8-way compass heading from centroid history (DirectionClassifier)
│   ├── live_tracking.py      # LiveSamTracker.step(): per-frame SAM 3.1 tracking with cached backbone
│   ├── model_host.py         # Refcounted MLX checkpoint residency host (HOST)
│   ├── vlm_registry.py       # Named local VLMs (gemma|lfm|lfm3b) for ask/report, loaded via model_host
│   ├── mlx_compat.py         # Idempotent shims so pinned mlx_vlm loads gemma4/LFM checkpoints (applied via apply_all())
│   ├── web_app.py            # FastAPI web UI (Aerial Ground Control)
│   ├── viz.py                # Set-of-Marks rendering, crop extraction
│   ├── agent_tools.py        # Agent-facing tools: ground_expression(), compute_relations()
│   ├── agent_loop.py         # VLM agent with tool loop
│   ├── references/           # system_prompt.txt for the agent
│   └── static/               # Web UI static assets
├── tests/
│   └── test_visionbrain.py   # All tests (pytest), organized by class
├── assets/samples/           # Test images and outputs
├── design-system/            # UI tokens/components; start at design-system/README.md
├── pyproject.toml            # Package metadata (setuptools, PEP 621)
├── SPEC.md                   # Detailed module-level API specification
└── CONTRIBUTING.md           # Contribution guide
```

## Build, Test, and Development Commands

The repo-local `.venv` (Python 3.14) has `mlx`, `mlx_vlm`, and `supervision` installed — prefer it:

```bash
# Run the full test suite (verified: passes without model weights; some tests skip)
.venv/bin/python -m pytest tests/ -v

# Run a single test class
.venv/bin/python -m pytest tests/test_visionbrain.py::TestCLI -v
```

Legacy alternative documented in README/CONTRIBUTING: the shared Poetry venv
(`FALCON_PY=~/Library/Caches/pypoetry/virtualenvs/falcon-perception-NVnkjaN--py3.12/bin/python`).
Any interpreter with `mlx` + `mlx_vlm` works; plain system `python` will not.

```bash
# Run the CLI directly (entry point: visionbrain.cli:main, also python -m visionbrain)
python -m visionbrain status
python -m visionbrain detect --image path/to/img.jpg --query "person"
```

No build step is required — the package uses `setuptools` with `src/` layout (`pip install -e .`).

CI (`.github/workflows/ci.yml`) runs pytest on Ubuntu with Python 3.12/3.13 after
`pip install -e ".[dev]"` (falling back to plain `pip install -e .`) — so tests
must pass with no MLX hardware and no cached weights.

## Coding Style & Conventions

- **Python 3.12+** with `from __future__ import annotations` in all modules
- **4-space indentation**, no trailing whitespace
- **Docstrings** on all public functions (Google-style brief descriptions)
- **Type hints** on function signatures
- **Imports**: stdlib first, then third-party, then local (`from .module import func`)
- **No external network calls at runtime** — all model weights come from the local Hugging Face cache
- **Domain-neutral copy** — the product is aerial/camera vision AI. No agriculture-specific wording in user-facing strings, defaults, placeholders, or model prompts (no cattle/pasture/farm examples; reports address an "operator" or "site manager"). Livestock terms may exist inside `prompt_router.py` vocabulary as routing capability only.
- **Read-only on upstream repos** — VisionBrain imports from Falcon-Perception but never modifies it
- **Graceful degradation** — if MLX or weights are missing, raise clear errors with actionable messages

## Testing Guidelines

- Framework: **pytest** (version >= 8.0)
- All tests live in `tests/test_visionbrain.py` organized into classes:
  `TestLoader`, `TestFalconPerception`, `TestAgentTools`, `TestViz`,
  `TestReviewOutputs`, `TestCLI`, `TestWebApp`, `TestDetectionCore`,
  `TestModelHost`, `TestVLMRegistry`, `TestLiveTracking`, `TestMlxCompat`
- Tests must pass without MLX hardware or cached model weights — heavy inference paths are skipped/mocked
- CLI smoke tests verify each `cmd_*` function handles missing arguments gracefully
- Loader tests validate model registry records and cache paths
- Run with: `.venv/bin/python -m pytest tests/ -v`

## Gotchas

- `supervision` **is** now a declared dependency (`supervision>=0.28,<0.30` — keep the `<0.30` pin). The heavy deps still *not* declared are `mlx` and `mlx_vlm`: `pyproject.toml` expects them pre-installed in the environment, detected at runtime with graceful fallback
- `visionbrain fastscan` is implemented by `cmd_fastscan()` in `cli.py`; `frame_selector.py` only provides the `score_frames()` scorer
- `src/visionbrain/__init__.py` exports `__version__` plus a single re-export, `DirectionClassifier` — do not rely on package-level re-exports of inference functions
- **Keep `detection_core.py` and `direction_tracking.py` dependency-free by design** (pure Python / numpy only, no MLX): the field bridge imports the `visionbrain` package directly on hosts with no models, so these must stay lightweight and importable anywhere
- `mlx_vlm` version window matters: `>= 0.6.1` (Gemma 4 KV-sharing weights) and `< 0.6.4` (0.6.4 drops SAM 3.1 support)
- `FAST_PIPELINE_SPEC.md` describes an in-progress feature (fast path + adaptive sampling); check status before assuming its behavior exists
- Web UI layout: three top-level tabs — **analyze** (video), **inspect** (image), **live** (field hub) — with all form controls in the right-rail **MISSION SETUP** panel (`#mission-setup` in `static/index.html`); tab switching toggles `.cfg-pane` elements by ID (`cfg-analyze` / `cfg-inspect` / `cfg-live`). Video modes (mission · track · fastscan) switch via `#c-mode`; image tasks (auto · detect · segment · sam3 · ocr, auto is keyword-routed) via `#c-task`. There is no bottom config bar. The analyze pipeline derives SAM targets from the query server-side (`prompt_router`) — there is no separate prompts field on the video pane
- Live tab talks **directly to the field hub** at `ws://127.0.0.1:8765` as role `dashboard` (observer relay ~5fps): binary frames use the bridge wire format `>III` = (frame_id, timestamp_ms, jpeg_len) + JPEG + `>I` telem_len + telemetry JSON — jpeg_len is at offset 8, pixels at offset 12; detections arrive as `type:"detections"` with an `items` array of normalized xyxy boxes. Controls send `set_engine` (incl. `lfm_model`: lfm|lfm3b), `set_vlm`, and `ask {question}` — see `LIVE` state object in index.html
- Live-tab **record** exports the annotated view client-side via `canvas.captureStream` + MediaRecorder (MP4 if supported, else WebM) — overlays included by construction since the canvas is what's recorded; **report** button posts `{type:"report", report_type:"field", summary}` where summary is a counts line built from the last detections frame
- The annotator palette constant is `SOM_PALETTE` in `viz.py` (renamed from `FARM_PALETTE`)
- SAM 3.1 weights (`mlx-community/sam3.1-bf16`, snapshot `a992e302…`) are already MLX-layout/post-sanitize; the local `.venv` patch at `.venv/lib/python3.14/site-packages/mlx_vlm/models/sam3_1/sam3_1.py::sanitize()` detects this (`mask_embed.conv` marker) and passes them through — without it every Conv2d double-transposes and `track_video` dies with a shape mismatch. Pre-patch original kept beside it as `sam3_1.py.bak-pre-fix`. If mlx-vlm is ever upgraded, re-check this guard still applies
- `prototype/` (repo root, untracked) holds standalone gallery/cockpit HTML design mockups; kept out of `static/` so the web app does not serve them

## Commit & Pull Request Guidelines

- Commit messages use **imperative mood** with a short summary (e.g., "Add SAM 3.1 JSON tracking + remote Gemma 4 reasoning")
- Keep `SPEC.md` in sync with any API changes
- New CLI commands require: a `cmd_<name>` function in `cli.py`, a registered subparser in `main()`, and a smoke test in `TestCLI`
- New inference modules require: module under `src/visionbrain/`, unit tests, and SPEC.md documentation
- Reference docs: `SPEC.md` (module APIs), `FAST_PIPELINE_SPEC.md` + `IMPLEMENTATION_NOTES*.md` (video pipeline design history), `design-system/README.md` (UI)

## Key Design Decisions

- **Two-machine architecture, evolved into a backend chain**: SAM 3.1 always runs locally (Mac Mini M4, 16GB); ask/report Gemma resolves a backend via `gemma_inference.available_backend()` — `ollama` (local endpoint) | `remote` (GPU server over HTTP) | `local` (MLX VLM via `vlm_registry`). Local VLM checkpoints (gemma/lfm/lfm3b) load through `model_host.HOST` so co-resident callers share one copy
- **MLX ecosystem**: All models use MLX-community weights for Apple Silicon optimization
- **FastAPI web UI**: The `web_app.py` module serves the VisionBrain — Aerial Ground Control dashboard (FastAPI title and `static/index.html` `<title>`) with static assets from `static/`
- **No modifications to existing projects**: VisionBrain reads from cached weights and the Falcon-Perception git repo without writing back
