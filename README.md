# VisionBrain

Agricultural vision AI on Apple Silicon — powered by Falcon Perception and SAM 3.1.

> Built to run locally. Reads weights from your Hugging Face cache. Uses MLX on Apple Silicon.

---

## Status

```
visionbrain status
```

```
=== VisionBrain Model Status ===
  [READY] tiiuae/Falcon-Perception
         Cache: ~/.cache/huggingface/hub/models--tiiuae--Falcon-Perception
         Size:  4.73 GB
         3B params, MLX, float16. Ready to use.
  [MISSING] facebook/sam3.1
         Run: huggingface-cli download facebook/sam3.1
```

---

## Setup

### Prerequisites

- **macOS** with Apple Silicon (M-series chip)
- **MLX** installed in your Python environment
- **Falcon Perception** weights cached locally (~4.7 GB)
- **SAM 3.1** weights cached locally (~3 GB, optional — needed for video tracking)

### Install

```bash
cd VisionBrain
pip install -e .
```

### Download model weights

```bash
# Falcon Perception (required)
huggingface-cli download tiiuae/Falcon-Perception

# SAM 3.1 (optional — video tracking only)
huggingface-cli download facebook/sam3.1
```

The VisionBrain Python environment must be the same one where MLX is installed (e.g. the `falcon-perception` Poetry venv, or a custom env with `mlx`, `mlx-vlm`, `transformers`, `pillow`, `pycocotools`, `opencv-python`).

---

## Usage

### Detect objects (fast — bounding boxes only)

```bash
visionbrain detect --image photo.jpg --query "cow"
visionbrain detect --image drone.jpg --query "sheep" --max-tokens 200
```

Output:
```
Detected 12 'cow' objects
  Preprocess: 7ms | Generation: 3708ms | Total: 3715ms
  Prefill tokens: 1416 | Decoded: 27 | Speed: 7.3 tok/s
  [1] score=1.000  cx=0.137 cy=0.718  h=0.488 w=0.274
  [2] score=1.000  cx=0.222 cy=0.722  h=0.498 w=0.387
  ...
```

### Segment objects (pixel-accurate masks)

```bash
visionbrain segment --image photo.jpg --query "cow" --output som.jpg
visionbrain segment --image photo.jpg --query "lame sheep" --output som.jpg
```

Saves a **Set-of-Marks** image with colored, numbered masks.

### Count objects in a crowd

```bash
visionbrain detect --image crowd.jpg --query "person" --max-tokens 200
# → Detected 36 'person' objects in 12.3s
```

### Read text from an image (OCR)

```bash
visionbrain ocr --image ear_tag.jpg
visionbrain ocr --image sign.jpg --question "read the brand name"
```

### SAM 3.1 multi-prompt detection (requires SAM 3.1 weights)

```bash
visionbrain sam3 \
  --image farm.jpg \
  --prompts cow sheep fence_post water_trough \
  --task detect
```

### Track objects in video (requires SAM 3.1 weights)

```bash
visionbrain track \
  --video pasture_cam.mp4 \
  --prompts cow horse \
  --output tracked.mp4
```

### VLM-powered visual reasoning agent

```bash
export OPENAI_API_KEY=sk-...
visionbrain agent \
  --image photo.jpg \
  --question "Which animal is showing signs of lameness?"
```

The agent loops: Falcon Perception for grounding → VLM for reasoning → until `answer()`.

---

## Architecture

```
visionbrain/
├── loader.py          Model registry — cache status, availability checks
├── fp_inference.py    Falcon Perception: segment(), detect(), ocr()
├── sam3_inference.py  SAM 3.1: detect_multi(), track_video(), track_realtime()
├── viz.py             Set-of-Marks rendering, crop extraction, relations
├── agent_tools.py     Agent-facing tools: ground_expression, compute_relations
├── agent_loop.py      VLM agent loop with context pruning
├── cli.py             CLI commands
└── references/        System prompt
```

**Key design decisions:**

- Read-only on Falcon-Perception repo — imports and calls, never modifies
- Caches MLX model in process memory — load once, use many times
- OCR decodes special markup tokens (`<|coord|><|size|><|seg|>`) — reflects Falcon Perception's coordinate output format
- Context pruning keeps agent history compact — max ~4 tool-call rounds per VLM call
- SAM 3.1 wraps `mlx_vlm` — requires `facebook/sam3.1` weights in Hugging Face cache

---

## Tests

```bash
# From VisionBrain/ (uses the Falcon Perception Poetry venv):
FALCON_PY=~/Library/Caches/pypoetry/virtualenvs/falcon-perception-NVnkjaN--py3.12/bin/python
$FALCON_PY -m pytest tests/ -v
```

---

## Adding a new CLI command

1. Add a `cmd_<name>` function in `cli.py`
2. Register the subparser in `main()`
3. Add a smoke test in `tests/test_visionbrain.py::TestCLI`
