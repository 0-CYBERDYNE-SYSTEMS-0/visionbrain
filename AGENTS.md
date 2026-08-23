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
  `TestReviewOutputs`, `TestCLI`, `TestWebApp`
- Tests must pass without MLX hardware or cached model weights — heavy inference paths are skipped/mocked
- CLI smoke tests verify each `cmd_*` function handles missing arguments gracefully
- Loader tests validate model registry records and cache paths
- Run with: `.venv/bin/python -m pytest tests/ -v`

## Gotchas

- `supervision_bridge.py` imports `supervision` at module top level, but `supervision` is **not** in `pyproject.toml` dependencies — it must be pre-installed in the environment (it is in `.venv`)
- `cmd_fastscan` exists twice: the **live** one is in `cli.py` (registered in the `main()` dispatch table); `frame_selector.py` holds a stale duplicate that is never called — change CLI behavior in the `cli.py` copy
- `src/visionbrain/__init__.py` currently exports only `__version__` — do not rely on package-level re-exports of inference functions
- `FAST_PIPELINE_SPEC.md` describes an in-progress feature (fast path + adaptive sampling); check status before assuming its behavior exists
- Web UI layout: all form controls live in the right-rail **MISSION SETUP** panel (`#mission-setup` in `static/index.html`) — there is no bottom config bar; tab switching toggles `.cfg-pane` elements by ID (`cfg-<tab>`)
- The annotator palette constant is `SOM_PALETTE` in `viz.py` (renamed from `FARM_PALETTE`)
- `pyproject.toml` metadata still carries legacy agricultural branding (description "Agricultural vision intelligence … livestock, crops", authors "FarmFriend / Cyberdyne Systems") — the domain-neutral copy rule applies to code/UI strings; package metadata is pending rebrand, so don't propagate wording in either direction

## Commit & Pull Request Guidelines

- Commit messages use **imperative mood** with a short summary (e.g., "Add SAM 3.1 JSON tracking + remote Gemma 4 reasoning")
- Keep `SPEC.md` in sync with any API changes
- New CLI commands require: a `cmd_<name>` function in `cli.py`, a registered subparser in `main()`, and a smoke test in `TestCLI`
- New inference modules require: module under `src/visionbrain/`, unit tests, and SPEC.md documentation
- Reference docs: `SPEC.md` (module APIs), `FAST_PIPELINE_SPEC.md` + `IMPLEMENTATION_NOTES*.md` (video pipeline design history), `design-system/README.md` (UI)

## Key Design Decisions

- **Two-machine architecture**: SAM 3.1 runs locally on Mac Mini M4 (16GB), Gemma 4 26B runs on remote GPU server via HTTP
- **MLX ecosystem**: All models use MLX-community weights for Apple Silicon optimization
- **FastAPI web UI**: The `web_app.py` module serves the VisionBrain — Aerial Ground Control dashboard (FastAPI title and `static/index.html` `<title>`) with static assets from `static/`
- **No modifications to existing projects**: VisionBrain reads from cached weights and the Falcon-Perception git repo without writing back
