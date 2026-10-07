# Live Painted-Mask Tracking — Demo Assets (2026-09-05)

Everything here was captured from the real application running on Apple Silicon
(MLX, sam3.1-bf16): no mockups, no composited overlays. The colored shapes on
each object are SAM 3.1's actual per-object masks, extracted per frame and
painted by the Ground Control console (live tab, local engine) or the mission
pipeline (analyze tab).

## Videos (`marketing/demo/video/`)

| File | What it shows |
|---|---|
| `live_masks_container_truck.mp4` | Container yard, top-down aerial. Two working vehicles tracked and painted simultaneously (`truck 77% · west`, `truck 58% · west`) — the label carries each object's live compass heading. ~80 s, 1280×720, the Ground Control live canvas as recorded in-browser. |
| `live_masks_marina_boats.mp4` | Marina pier, top-down aerial. Sixteen distinct hulls each locked to its own color-coded painted mask (scores 0.70–0.95, stable track IDs). The marquee "one object stays painted" shot. ~27 s, 1280×720. |
| `*_analyzed.mp4` (see job names below) | Server-side mission renders (analyze tab, FULL MISSION, SAM-only): every frame annotated at source resolution. |

Mission renders produced today:

- `e16d8743fda1_analyzed.mp4` — container yard (cropped), `truck`, threshold 0.45, every frame
- harbor + factory renders land as `9c9f16dd4609_analyzed.mp4` / (factory job) `_analyzed.mp4` in the web app results dir; copies belong in this folder — see `marketing/demo/README.md` for the canonical naming.

## Stills (`marketing/shots/`)

- `live_masks_marina_hero_1280x720.png` — **hero image**: 16 painted hulls, one color per track
- `live_masks_marina_alt_1280x720.png` — late-clip alt frame
- `live_masks_container_trucks_1280x720.png` — two painted trucks + heading labels

## Capabilities demonstrated (and how to read the frame)

1. **Painted mask tracking** — the fill on each object is SAM 3.1's mask outline
   (`detection_core.mask_to_polygon`), not a bounding box. Boxes appear only as
   a fallback when an item carries no mask.
2. **Identity persistence** — each object keeps its color + track ID across
   frames (palette is keyed by `track_id`).
3. **Direction awareness** — the `· west` / `· east` suffix in each label is the
   per-track 8-way compass heading (`direction_tracking.DirectionClassifier`),
   computed from centroid motion.
4. **Selective prompting** — one text prompt ("truck", "boat") isolates the
   objects that matter; confidence is shown per object. The demo clips were
   deliberately chosen so only a handful of real objects match — that is the
   intended operating point (threshold ~0.4–0.5), not dozens of sub-threshold
   hits.

## Honest scope notes

- Distances/speeds: target-level metric speed is NOT claimed in these clips.
  The telemetry panel carries platform altitude/ground speed/GPS; per-object
  heading is live. GSD-based per-object speed is a natural next step.
- Falcon-OCR remains registry-only in VisionBrain (upstream serves it via
  vLLM/CUDA); it is downloaded but not wired into MLX inference.
- Track IDs can churn on fast aerial orbits with broad prompts; the marina and
  yard clips hold stable IDs.

## Footage attribution (Pexels License, free to use)

- Container yard: pexels.com/video/8783391 (top-down container terminal, HD 1080p)
- Marina: pexels.com/video/4813259 (top-down fishing boats at pier, HD 1080p)
- Factory: pexels.com/video/12786489 (aerial industrial plant, HD 1080p)

Clips were trimmed/cropped (ffmpeg) for framing; the container-yard close-up is
a 2× center crop of the source so the working truck reads at SAM's resolution.

## Reproduce

```bash
.venv/bin/python -m visionbrain ui --no-browser   # http://127.0.0.1:7860
# live tab → local engine → load a clip → prompt "truck"/"boat" → threshold ~0.45 → start
# analyze tab → FULL MISSION, FALCON+ off, EVERY=1F for fully-annotated renders
```
