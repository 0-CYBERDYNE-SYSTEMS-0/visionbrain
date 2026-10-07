# creative brief — 60-second VisionBrain client demo video

Write a 60-second demo video script that shows potential clients what
VisionBrain does and how it makes money for them. Everything below is
source-checked; the validator will hold the script to it.

## product (what it is)

VisionBrain is an aerial & camera vision AI workstation that runs on a
customer's own Apple Silicon Mac (via MLX). Three models work as one
pipeline from a single natural-language query:

| stage     | model             | job                                              |
|-----------|-------------------|--------------------------------------------------|
| TRACKING  | SAM 3.1           | per-frame masks, persistent track ids            |
| GROUNDING | Falcon Perception | expression-level detect · segment · ocr          |
| REASONING | Gemma 4           | field reports, anomaly calls, plain-language q&a |

one query in → annotated video + structured json + a written field report out.
the customer's footage never leaves their machine. no cloud. no subscription.

the product surface is "Ground Control", a web ui with three panes:
- ANALYZE — recorded footage missions (track / fastscan)
- INSPECT — single-image detect / segment / ocr
- LIVE    — field hub: live relay, tracking, zones, direction, evidence clips, ask-the-scene q&a, field reports

## the monetizable pipelines (what the video must sell)

source: BUSINESS_CAPABILITY_RECON.md (2026-09-09). the reusable value is
"the workflow connecting an observation to structured results and a
business action." three sellable pipeline stories:

1. RECORDED-FOOTAGE REVIEW — point a query at recorded aerial video; get
   tracked objects, counts, and a written report before the crew lands.
   verticals: ports & terminals, logistics yards, corridors, shipyards.
2. LIVE SITE MONITORING — a live camera or drone feed watched by the
   engine: zone counters (line + polygon), 8-way direction, dwell, and
   automatic evidence clips when a trigger fires. "a fence that writes
   its own log."
3. OBSERVATION → BUSINESS ACTION — the operator asks the scene a question
   in plain language and gets an answer, a written field report, and
   preserved evidence to attach to a work order or incident record.

## audience

potential clients — port/terminal ops managers, security & site
supervisors, drone-service providers. not developers. no code shown in
the copy; the ui itself may appear.

## voice (marketing/copy.md — mandatory)

- senior field engineer writing for another one. precise, plain, faintly dry.
- numbers carry the argument.
- no market-speak, no emoji, no rocket ships, no hype adjectives
  ("revolutionary", "powerful", "cutting-edge" are all banned).
- on-screen labels small caps; on-screen body lowercase. sentence case headings.
- approved numbers you may use (real, measured on this machine):
  - 480 frames · 96 processed · 338 unique objects tracked — one query
  - ~3.7 fps end-to-end tracking at 512 px, every 5th frame, apple silicon
  - falcon grounding ~0.4 s warm per image
  - three models co-resident on one 32 gb mac
  - 0 bytes of footage leaving the machine
- approved closing line: "bring a 20-second clip. leave with a report."

## hard honesty constraints (from the recon — validator enforces)

1. one live source per instance. never imply multi-camera or multi-site
   dashboards ("watch every site" is banned).
2. live detection is sam-based; don't claim other live detectors.
3. direction is 8-way in image coordinates — say "direction"/"heading",
   never "georeferenced" or "gps heading".
4. dwell = stationary duration. don't claim time-in-named-zone analytics.
5. no accuracy claims ("99% accurate", "ai-powered precision" banned).
6. privacy claim allowed in this exact shape: footage/results stay on the
   customer's machine. don't claim certifications or compliance.
7. zones exist as line + polygon counters; direction triggers and evidence
   clips are real (35 clips in one session is documented).

## timing budget (hard)

- total: exactly 60.0 s across 6 scenes.
- scene durations must be stated and must sum to 60.0.
- narration pace: ≤ 2.4 words per second. per scene: vo word count ≤
  duration × 2.2 (leave breathing room). total vo ≤ 132 words.
- scenes with no vo are allowed but total silent time ≤ 8 s.

## asset inventory (reference these in the visuals column)

real footage (marketing/demo/):
- video/rotterdam_analyzed.mp4 — 1920×1012, 20 s. annotated batch output:
  boats tracked over the port of rotterdam. hook material.
- video/mission_marina_boats_everyframe.mp4 — 1920×1080, 10.4 s. analyze
  tab mission: painted masks, boats, marina.
- video/mission_containers_truck_everyframe.mp4 — 1280×720, 12.9 s.
  analyze mission: container yard, trucks.
- video/live_masks_container_truck.mp4 — 1280×720, 81 s. live tab: painted
  masks on container trucks, persistent ids. live-scene material.
- video/live_masks_marina_boats.mp4 — 1280×720, 26.6 s. live tab: marina.
- video/live_harbor_tracking.mp4 — 1280×720, 20 s. live tracking canvas,
  hand-drawn target followed with compass heading.

dashboard stills, 1920×1080 (marketing/shots/):
- analyze_setup_1920.png, analyze_running_1920.png, analyze_results_1920.png,
  analyze_report_1920.png (the written report on screen)
- inspect_detect_1920.png, inspect_segment_1920.png
- live_terminal_1920.png, live_quay_1920.png
- mission_marina_masks_1920x1080.png
- annotated_frame.jpg

## required output format

a markdown script with:
1. title + total runtime line.
2. one block per scene, each with: scene id, TC in–out, duration,
   visuals (asset path + motion note), on-screen text (exact strings,
   lowercase body / small-caps labels), VO (exact narration words),
   and rationale in one line.
3. a totals block: scene durations summing to 60.0, total vo word count,
   and a check that every claim traces to this brief.

write the script to: marketing/demo60/script_v1.md
