# VisionBrain — Live Capability Demo (2026-09-04)

Everything here was captured from the real app driving real models on this
machine (Apple M2 Pro, 32 GB) — SAM 3.1 tracking, direction triggers, and
smart capture running on aerial footage of the Port of Rotterdam
(`assets/samples/rotterdam_1080p.mp4`).

## The money shot

- `shots/05_live_canvas_boxes.png` — live SAM 3.1 tracking over the harbor:
  a hand-drawn `target 1 · northeast` box being followed with a compass
  heading, plus autonomous `boat · east` tracks, all labeled live at ~224 px
  realtime resolution in the browser.
- `shots/06_live_canvas_boxes_b.png` — same session, 6 s later (tracks moved).

## Video

- `video/live_harbor_tracking.mp4` — 20 s of the live tracking canvas
  (5 fps capture of the actual dashboard canvas): boats tracked with
  direction headings while the harbor loop plays.
- `video/target_draw_live.mp4` — 18 s; at ~4 s a new target box is drawn
  onto a detected boat mid-stream and the engine starts tracking it.
- `smart_capture_clips/clip_*.mp4` — server-side auto-capture clips: the
  direction trigger ("any heading") fired and the engine rolled evidence
  clips on its own (35 were captured in one session; 3 sampled here). Clips
  are clean footage by design — overlays are a client render layer.

## Batch pipeline stills

- `shots/02_mission_tracking.png` — FULL MISSION progress UI during SAM
  tracking (open-vocab routing: query "boats and people near the pier" →
  SAM targets `boats, people, pier`).

## How it was driven

`python -m visionbrain ui` → browser → LIVE tab → LOCAL ENGINE → rotterdam
loop → prompts `boat person` → draw target → arm direction trigger. Server
API used for uploads only. Frames exported from the live canvas at 1280×720.

## What footage would level this up

1. 30–60 s aerial/drone harbor clips, 1080p+, boats AND people moving,
   static or slow-panning camera (tracking reads best without fast motion).
2. A dock/pier scene with identifiable vessel names — showcases the
   SAM-track → crop → Falcon-OCR hull-name pipeline.
3. A real RTSP camera feed — makes the live tab a true live demo.
4. Night or thermal IR footage — shows the frontier edge of the tracker.

---
## 2026-09-05 addendum — painted-mask demo set

A dedicated set of live painted-mask tracking assets (container yard, marina,
factory) plus stills now lives in [`video/LIVE_MASKS_README.md`](video/LIVE_MASKS_README.md).
Videos: `video/live_masks_container_truck.mp4`, `video/live_masks_marina_boats.mp4`,
`video/mission_*_everyframe.mp4`. Stills: `../shots/live_masks_*.png`,
`../shots/mission_*.png`.
