# VisionBrain Fast Pipeline — Implementation Spec

> Fast-path + adaptive sampling for large video analysis.
> Status: **IN PROGRESS**

---

## Goals

1. **Instant first impression** — sub-60s Falcon-only scan at low res before full pipeline starts
2. **Adaptive SAM** — skip static frames using motion delta, propagate masks temporally
3. **Parallel Falcon** — process key frames concurrently using ThreadPoolExecutor
4. **Graceful UX** — user gets a quick answer fast, full analysis continues in background

---

## Architecture

```
Upload video
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  FAST-PATH (runs immediately, < 60s)                       │
│                                                             │
│  1. Extract frames at 360p every 5 seconds                  │
│  2. Falcon zero-shot: "Is <query> present? Where?"         │
│  3. Return: quick answer + temporal regions of interest    │
│  4. (Optional) Launch full pipeline in background           │
└─────────────────────────────────────────────────────────────┘
    │
    ▼ (if full analysis requested)
┌─────────────────────────────────────────────────────────────┐
│  FULL PIPELINE                                              │
│                                                             │
│  SAM 3.1 adaptive:                                          │
│    - Motion-guided frame skipping (delta threshold)          │
│    - Temporal mask propagation (backbone every N,           │
│      propagate forward ±P frames)                            │
│    - Query-guided relevance filter (skip irrelevant frames) │
│                                                             │
│  Falcon parallel refinement:                                 │
│    - ThreadPoolExecutor over key frames                     │
│    - All Falcon calls run concurrently                       │
│                                                             │
│  Gemma 4 reasoning → field report                           │
└─────────────────────────────────────────────────────────────┘
```

---

## New Module: `frame_selector.py`

### Purpose
Fast relevance filter and key-frame ranker using Falcon Perception at low resolution.

### API

```python
def score_frames(
    video_path: str,
    query: str,
    *,
    sample_every_n_seconds: float = 5.0,
    max_frames: int = 60,
    resolution: int = 360,
) -> FrameScores:
    """Score frames by likelihood of containing query-relevant content.

    Returns:
        FrameScores with:
        - scores: list of (frame_index, timestamp, relevance_score)
        - regions: list of (start_time, end_time) high-interest windows
        - quick_answer: str — "Yes, persons found at 0:12-0:45 and 1:30-2:10"
        - is_relevant: bool — True if query found anywhere
    """
```

### Algorithm

1. Extract frames at uniform intervals (default: every 5s, max 60 frames)
2. Run Falcon `detect()` on each at 360p
3. Score each frame: `relevance = detection_count / max_detections * 0.7 + text_match * 0.3`
4. Cluster high-scoring frames into temporal regions (gap > 10s = new region)
5. Build quick-answer string from regions
6. Return immediately

### CLI Integration

```
visionbrain fastscan --video <path> --query "person"
```

---

## Phase 1: Fast-Path (`--fast` flag on `analyze`)

### Changes

- `cli.py`: `cmd_analyze` gains `--fast` flag
- `frame_selector.py`: new module added
- `cmd_analyze` flow with `--fast`:
  1. Run `score_frames()` immediately
  2. Print quick answer to stdout + SSE
  3. If `--fast` only: exit here with results
  4. If full pipeline: spawn background job for SAM+Falcon+Gemma

### SSE Behavior

With `--fast`, the job emits structured events:
```
data: {"type":"quick_answer","text":"Persons detected 0:12-0:45 and 1:30-2:10","regions":[...]}
data: {"type":"done","status":"partial","results":{"quick_answer":"..."}}
```

---

## Phase 2: Adaptive SAM (`--adaptive` flag)

### New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--adaptive` | flag | False | Enable motion-guided skipping |
| `--motion-threshold` | float | 0.03 | Frame delta threshold (0-1, lower = more sensitive) |
| `--propagate` | int | 0 | Propagate masks forward N frames (0 = off) |
| `--relevance-filter` | flag | False | Skip frames Falcon says are irrelevant |

### Implementation

In `sam3_inference.py`, add to `track_video_with_json`:

```python
def track_video_with_json(
    ...
    adaptive_motion: bool = False,
    motion_threshold: float = 0.03,
    propagate_frames: int = 0,
    relevance_scores: dict[int, float] = None,  # frame_idx → score
    relevance_threshold: float = 0.2,
) -> tuple[VideoTrackStats, list[dict]]
```

#### Motion-Guided Skipping

```python
prev_gray = None
for fi in range(total_frames):
    ret, frame_bgr = cap.read()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if prev_gray is not None:
        delta = np.abs(gray.astype(float) - prev_gray.astype(float)).mean() / 255.0
        if delta < motion_threshold and fi % every_n_frames != 0:
            # Skip expensive processing, use propagated result
            continue
    prev_gray = gray
    # ... normal processing
```

#### Temporal Mask Propagation

```python
# When backbone runs at frame F, propagate masks to F+1 ... F+propagate_frames
# without re-running ViT. Use SimpleTracker's temporal propagation.
if propagate_frames > 0:
    propagated_masks = tracker.propagate_masks(latest_result, propagate_frames)
    # Apply propagated masks to next N frames
```

#### Relevance-Filtered Skipping

```python
# Skip frames where Falcon scored relevance below threshold
if relevance_scores is not None:
    score = relevance_scores.get(fi, 1.0)
    if score < relevance_threshold and fi % every_n_frames != 0:
        continue  # skip expensive processing
```

---

## Phase 3: Parallel Falcon (`--falcon-parallel` flag)

### Changes

In `cli.py`, `cmd_analyze`:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def _refine_falcon_frame(args_tuple):
    fi, ts, pil_frame, query = args_tuple
    from .fp_inference import detect
    results, stats = detect(pil_frame, query, max_new_tokens=200)
    return fi, ts, results, stats

# Replace sequential loop with:
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(_refine_falcon_frame, (fi, ts, pf, args.query)): (fi, ts)
        for fi, ts, pf in key_frames
    }
    for future in as_completed(futures):
        fi, ts, results, stats = future.result()
        # ... collect results
```

### Thread Safety

- Falcon model is re-entrant (per-call load, no global state)
- ThreadPoolExecutor with 4 workers max to avoid memory pressure
- Results collected and merged after all futures complete

---

## CLI Changes

### New Flags on `analyze`

```
--fast                     Run fast-path Falcon scan first, return immediately
--adaptive                 Enable adaptive SAM (motion-guided + propagation)
--motion-threshold FLOAT   Frame delta threshold for motion skip (default 0.03)
--propagate INT            Propagate masks forward N frames (default 0=off)
--relevance-filter         Skip frames Falcon says are irrelevant
--parallel-falcon          Process Falcon key-frames in parallel (default True)
--quick-output PATH        Write fast-path result to this file
```

### New Command

```
visionbrain fastscan --video <path> --query <expr>
```

---

## Implementation Notes

- **Decision 1**: `track_realtime` already has `detect_every` and `recompute_backbone_every` — reusing this pattern for `track_video_with_json` with `every_n_frames` and `backbone_every` felt cleaner than adding a new function. The adaptive motion-skip is a new mode layered on top, not a replacement.
- **Decision 2**: Temporal mask propagation uses `SimpleTracker`'s existing mask propagation rather than implementing custom interpolation. If `propagate_frames` is set, we call a new `tracker.propagate_masks()` method.
- **Decision 3**: Parallel Falcon uses `ThreadPoolExecutor` (threads, not processes) because the GIL is released during MLX operations and we want shared memory for the model weights already loaded in the parent process.
- **Decision 4**: Fast-path is blocking — it runs and returns before any SAM processing starts. If the user wants background continuation, they pass `--background` or the web UI handles it by spawning two jobs.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/visionbrain/frame_selector.py` | **NEW** — fast Falcon frame scorer |
| `src/visionbrain/sam3_inference.py` | Add adaptive params + propagation to `track_video_with_json` |
| `src/visionbrain/cli.py` | Add `--fast`, `--adaptive`, `--propagate`, `--relevance-filter` flags; parallel Falcon |
| `src/visionbrain/__init__.py` | Export `frame_selector` |
| `web_app.py` | Handle `--fast` job type with SSE quick_answer events |
| `SPEC.md` | Add new API surface for fast-path and adaptive |
| `IMPLEMENTATION_NOTES_FAST_PIPELINE.md` | This file |

---

## Verification

```bash
# Fast scan only
.venv/bin/python -m visionbrain fastscan --video test.mp4 --query "person"

# Adaptive full pipeline
.venv/bin/python -m visionbrain analyze --video test.mp4 --query "person" \
  --fast --adaptive --propagate 5 --motion-threshold 0.03

# Compare timing
time .venv/bin/python -m visionbrain analyze --video test.mp4 --query "person" \
  --every 2 --backbone-every 1 --resolution 1008

time .venv/bin/python -m visionbrain analyze --video test.mp4 --query "person" \
  --adaptive --propagate 5 --motion-threshold 0.03 --every 5
```
