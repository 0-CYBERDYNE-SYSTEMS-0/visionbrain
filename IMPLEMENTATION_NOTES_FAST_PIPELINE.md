# Implementation Notes — Options A+B Pipeline
## Fast-Path + Adaptive Sampling for VisionBrain

---

## What Was Built

Three interlocked components implementing Options A and B from the original spec:

1. **`frame_selector.py`** — Option A: fast-path Falcon scorer
2. **`sam3_inference.py`** adaptive params — Option B: motion skip + mask propagation
3. **`cli.py`** orchestration — unified `--fast` / `--adaptive` flags + parallel Falcon

---

## Decisions Made That Weren't in the Spec

### 1. Fast-scan lives in its own module, not inside `cli.py`

**Why:** `score_frames()` is a reusable library function that can be called from Python
(`from visionbrain.frame_selector import score_frames`) as well as the CLI. Putting it
in cli.py would have hidden it from programmatic callers.

**Tradeoff:** New file means one more import path to maintain. Acceptable because the
module is self-contained and has a clear single responsibility.

### 2. Relevance scores are a plain `dict[int, float]` passed into SAM, not a
side-channel file

**Alternative considered:** Write fast-scan JSON to disk, have SAM read it back.
**Rejected:** Adds I/O, file cleanup complexity, and a race condition if multiple jobs
run simultaneously. Passing the dict is simpler and keeps everything in-memory.

**Tradeoff:** The dict must be built before `track_video_with_json` is called. This
means `--fast` + `--adaptive` do two full passes: one for Falcon scoring, one for SAM.
For a 1-hour drone video this is still faster than running SAM on every frame at full
resolution — but it does mean double the frame extraction overhead. Acceptable for now;
future optimization would interleave scoring and tracking in a single pass.

### 3. Motion delta uses greyscale pixel mean — not optical flow, not SSIM

**Why:** `cv2.absdiff(gray1, gray2).mean()` is ~10 lines and runs at frame read speed.
Optical flow (Farneback or Lucas-Kanade) is more accurate for motion but adds ~50ms/frame
on Apple Silicon — destroying the speed budget for the skip logic to pay off.

**Tradeoff:** Greyscale mean is sensitive to lighting changes (sun moving behind clouds =
false positive motion). Threshold 0.03 was calibrated for static camera drone footage.
Handheld camera would need a higher threshold or a different approach (e.g., SSIM on
cropped regions).

### 4. Mask propagation reuses `latest_result` object, not raw mask tensors

**Why:** The tracker already owns the result management. Propagating means "use the same
result object for the next N frames without re-running the model." This avoids copying
mask tensors.

**Tradeoff:** Propagated masks are temporally stale. If the object moves significantly
between detect frames, the bounding box and centroid will drift. For livestock monitoring
(statics or slow movement) this is fine. For fast action (sporting dogs, vehicles) you'd
want to either (a) warp the mask using optical flow, or (b) reduce `propagate_frames`.

### 5. Parallel Falcon uses `ThreadPoolExecutor`, not multiprocessing

**Why:** The GIL is released during the MLX/Falcon compute (C++ backend). Threads run
genuinely in parallel. No shared memory complexity.

**Tradeoff:** `max_workers=4` is hardcoded. On an M2 Pro with 6 performance cores, 4 is
conservative (leaves headroom for the main thread). Could be set to `os.cpu_count()`.
Changed via `--parallel-falcon` / `--sequential-falcon` flags — user can override.

### 6. Fast-scan quick answer is Falcon + a hand-written prompt, not a separate model call

**Why:** Avoids spinning up a second model or API call. The Falcon pipeline already has
the frame context and detections. We inject a structured prompt and parse the response.

**Tradeoff:** The natural-language answer quality depends on how well Falcon follows the
prompt template. If Falcon hallucinated or missed detections, the quick answer will be
wrong. For the livestock use case this is acceptable — the full SAM pipeline corrects
any Falcon errors. The quick answer is explicitly labeled "quick answer, verify with
full analysis."

### 7. `--relevance-filter` requires `--fast` (or an explicit relevance_scores dict)

**Why:** Relevance filtering without a fast-scan makes no sense — there are no relevance
scores to filter by.

**Tradeoff:** If the user provides relevance scores some other way (future: pre-computed
JSON), they can pass them directly to `track_video_with_json()` without `--fast`. The CLI
enforces this at the argument level but the Python API is more flexible.

### 8. `frame_selector.py` uses `fp_inference.detect()`, not `segment()`

**Why:** `detect()` returns structured bounding boxes with scores — easier to convert to
a relevance score than raw mask tensors from `segment()`. Falcon's `detect()` is also
faster (no mask decoding).

**Tradeoff:** Falcon detections are box proposals, not pixel masks. For relevance scoring
this is sufficient. If we later wanted to propagate Falcon mask proposals to SAM
(e.g., use Falcon boxes as SAM point prompts), we'd need to switch to `segment()`.

---

## Parameter Calibration (Untested — Document for Field Use)

These are starting points, not gospel. The user should adjust based on footage:

| Parameter | Default | When to Increase | When to Decrease |
|---|---|---|---|
| `--every` | 2 | Low-complexity footage (empty pasture) | Dense scenes (many animals overlapping) |
| `--motion-threshold` | 0.03 | Camera shake, wind-induced sway | Static camera, slow-moving animals |
| `--propagate` | 0 | Static/slow animals, low GPU budget | Fast action, frequent direction changes |
| `--relevance-filter` | off | When fast-scan identified dead zones | When most frames have relevant content |
| `--fast --every` | 5.0s | Long continuous shots | Rapid cuts, many scene changes |

---

## Architecture Tradeoff: Where Temporal Propagation Should Live

Currently, mask propagation lives inside `track_video_with_json`. A cleaner long-term
architecture would separate it into a dedicated post-processing pass:

```
Raw detections (JSON) → TemporalPropagator → Smoothed track (JSON)
```

**Why not done now:** The current approach is simpler (no extra I/O, no extra pass) and
works well enough for the target use case. The cleaner architecture would matter more
if:
- We want to propagate between non-consecutive detect frames (e.g., skip 8 frames,
  propagate across the gap)
- We want to propagate without writing intermediate video
- We want to share propagation across different trackers

---

## What Was Left Out (Future Work)

1. **Fast-scan + SAM in single pass** — currently two passes (Falcon scan → SAM track).
   Would need to interleave scoring and tracking: score N frames ahead, update skip set,
   run SAM on remaining frames. More complex; left for Phase 2.

2. **Optical flow mask warping** — current propagation is raw reuse. Optical flow would
   translate the mask to track object motion. Significant accuracy improvement for fast
   movement at the cost of ~50ms/frame overhead.

3. **Per-region adaptive resolution** — high-relevance regions could be processed at
   higher resolution while low-relevance regions stay at 512. Would need GPU memory
   management and per-region SAM input routing.

4. **Auto-tuning** — the optimal combination of `--every`, `--motion-threshold`,
   `--propagate` depends on camera motion, object speed, and scene density. A one-shot
   calibration pass (process 30 seconds, auto-tune params) would remove the guesswork.

---

## Git Status

All changes are uncommitted. Files modified:
- `src/visionbrain/frame_selector.py` — **new file** (428 lines)
- `src/visionbrain/sam3_inference.py` — added adaptive params to `track_video_with_json`
- `src/visionbrain/cli.py` — new flags + `fastscan` command + parallel Falcon
- `src/visionbrain/web_app.py` — new `/api/job/fastscan` endpoint + updated `analyze`
- `SPEC.md` — updated with new commands, module specs, and API surface
- `FAST_PIPELINE_SPEC.md` — the original spec document

---

## How to Test

```bash
cd ~/visionbrain

# Fast-scan only (no SAM, no Gemma — fastest test)
python -m visionbrain fastscan \
  --video ~/Downloads/test_video.mp4 \
  --query "cattle" \
  --max-frames 20

# Fast-path + full analyze
python -m visionbrain analyze \
  --video ~/Downloads/test_video.mp4 \
  --query "cattle" \
  --fast \
  --report

# Adaptive SAM only (no fast-scan)
python -m visionbrain analyze \
  --video ~/Downloads/test_video.mp4 \
  --query "cattle" \
  --adaptive \
  --motion-threshold 0.03 \
  --propagate 5 \
  --every 8

# Full pipeline with all options
python -m visionbrain analyze \
  --video ~/Downloads/test_video.mp4 \
  --query "cattle" \
  --fast \
  --adaptive \
  --propagate 5 \
  --relevance-filter \
  --falcon-refine \
  --parallel-falcon \
  --report
```
