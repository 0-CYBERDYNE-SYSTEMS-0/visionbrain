# Repository Guidelines

## Project Overview

VisionBrain is an agricultural vision AI toolkit running on Apple Silicon via MLX. It provides a Python library and CLI for three models — Falcon Perception (segmentation/detection/OCR), SAM 3.1 (video tracking), and Gemma 4 26B (reasoning/reports) — orchestrated in a two-machine pipeline where SAM runs locally and Gemma runs on a remote GPU server.

## Project Structure

```
VisionBrain/
├── src/visionbrain/          # All source code
│   ├── cli.py                # CLI entry point (argparse subcommands)
│   ├── loader.py             # Model registry and Hugging Face cache status
│   ├── fp_inference.py       # Falcon Perception: segment(), detect(), ocr()
│   ├── sam3_inference.py     # SAM 3.1: detect_multi(), track_video(), track_realtime()
│   ├── gemma_inference.py    # Gemma 4: ask(), generate_report()
│   ├── remote_gemma_inference.py  # Remote Gemma 4 via HTTP
│   ├── web_app.py            # FastAPI web UI (TERRA-VISION Ground Control)
│   ├── viz.py                # Set-of-Marks rendering, crop extraction
│   ├── agent_tools.py        # Agent-facing tools: ground_expression(), compute_relations()
│   ├── agent_loop.py         # VLM agent with tool loop
│   └── static/               # Web UI static assets
├── tests/
│   └── test_visionbrain.py   # All tests (pytest)
├── assets/samples/           # Test images and outputs
├── pyproject.toml            # Package metadata (setuptools, PEP 621)
├── SPEC.md                   # Detailed module-level API specification
└── CONTRIBUTING.md           # Contribution guide
```

## Build, Test, and Development Commands

Tests require the shared Poetry environment with `mlx` and `mlx_vlm` pre-installed:

```bash
# Run the full test suite
FALCON_PY=~/Library/Caches/pypoetry/virtualenvs/falcon-perception-NVnkjaN--py3.12/bin/python
$FALCON_PY -m pytest tests/ -v

# Run the CLI directly
python -m visionbrain status
python -m visionbrain detect --image path/to/img.jpg --query "cattle"
```

No build step is required — the package uses `setuptools` with `src/` layout.

## Coding Style & Conventions

- **Python 3.12+** with `from __future__ import annotations` in all modules
- **4-space indentation**, no trailing whitespace
- **Docstrings** on all public functions (Google-style brief descriptions)
- **Type hints** on function signatures
- **Imports**: stdlib first, then third-party, then local (`from .module import func`)
- **No external network calls at runtime** — all model weights come from the local Hugging Face cache
- **Read-only on upstream repos** — VisionBrain imports from Falcon-Perception but never modifies it
- **Graceful degradation** — if MLX or weights are missing, raise clear errors with actionable messages

## Testing Guidelines

- Framework: **pytest** (version >= 8.0)
- All tests live in `tests/test_visionbrain.py` organized into classes (`TestLoader`, `TestCLI`, `TestViz`, etc.)
- CLI smoke tests verify each `cmd_*` function handles missing arguments gracefully
- Loader tests validate model registry records and cache paths
- Run with: `$FALCON_PY -m pytest tests/ -v`

## Commit & Pull Request Guidelines

- Commit messages use **imperative mood** with a short summary (e.g., "Add SAM 3.1 JSON tracking + remote Gemma 4 reasoning")
- Keep `SPEC.md` in sync with any API changes
- New CLI commands require: a `cmd_<name>` function in `cli.py`, a registered subparser in `main()`, and a smoke test in `TestCLI`
- New inference modules require: module under `src/visionbrain/`, exports in `__init__.py`, unit tests, and SPEC.md documentation

## Key Design Decisions

- **Two-machine architecture**: SAM 3.1 runs locally on Mac Mini M4 (16GB), Gemma 4 26B runs on remote GPU server via HTTP
- **MLX ecosystem**: All models use MLX-community weights for Apple Silicon optimization
- **FastAPI web UI**: The `web_app.py` module serves the TERRA-VISION Ground Control dashboard with static assets from `static/`
- **No modifications to existing projects**: VisionBrain reads from cached weights and the Falcon-Perception git repo without writing back
