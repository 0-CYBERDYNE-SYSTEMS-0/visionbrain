# Contributing to VisionBrain

We welcome contributions! Here's how to run the test suite and what to know before submitting.

## Running Tests

VisionBrain uses **pytest**. The tests require the shared Poetry environment that has `mlx` and `mlx_vlm` pre-installed.

```bash
cd VisionBrain
FALCON_PY=~/Library/Caches/pypoetry/virtualenvs/falcon-perception-NVnkjaN--py3.12/bin/python
$FALCON_PY -m pytest tests/ -v
```

Or with Poetry:

```bash
cd VisionBrain
poetry run pytest tests/ -v
```

## Adding a New CLI Command

1. Add a `cmd_<name>` function in `src/visionbrain/cli.py`
2. Register the subparser in `main()`
3. Add a smoke test in `tests/test_visionbrain.py::TestCLI`

## Adding a New Inference Module

1. Add the module under `src/visionbrain/`
2. Export public functions in `src/visionbrain/__init__.py`
3. Add unit tests in `tests/`
4. Document the public API in `SPEC.md`

## Design Principles

- **Read-only on upstream repos** — VisionBrain imports from Falcon-Perception but never modifies it.
- **Graceful degradation** — If MLX or model weights are missing, functions should raise clear errors.
- **No external network calls** — All model weights come from the local Hugging Face cache.

## Submitting Changes

1. Fork the repo and create a feature branch.
2. Ensure all tests pass (`pytest tests/ -v`).
3. Keep `SPEC.md` in sync with any API changes.
