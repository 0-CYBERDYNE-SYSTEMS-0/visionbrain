# VisionBrain — marketing copy deck

Voice: senior field engineer writing for another one. Precise, plain, faintly
dry. Numbers carry the argument. No market-speak, no emoji, no rocket ships.
Headings sentence case; labels small caps; body lowercase.

---

## Positioning line (one-liner)

aerial intelligence that runs on your desk.
three vision models, one pipeline — tracking, grounding, reasoning —
on apple silicon. no cloud. no subscription. your footage never leaves the building.

## Alternative one-liners (A/B)

- the ground-control station for drone fleets that can't ship their footage to someone else's cloud.
- point a query at twenty minutes of aerial video. get tracked objects, counts, and a written report.
- sam 3.1 tracks it. falcon finds it. gemma explains it. your mac runs all three.

## The three-model pipeline (explainer block)

| stage    | model             | job                                              |
|----------|-------------------|--------------------------------------------------|
| TRACKING | SAM 3.1           | per-frame masks, persistent track ids            |
| GROUNDING| Falcon Perception | expression-level detect · segment · ocr          |
| REASONING| Gemma 4           | field reports, anomaly calls, plain-language q&a |

caption: one query in — annotated video, structured json, and a written field report out.

## Clear bullet points (the argument)

**runs where your data lives**
- everything executes on your hardware — apple silicon via MLX. footage, weights, results: all local.
- weights live in your hugging face cache. results live on your disk. no sync, no telemetry, no subscription.
- reasoning resolves automatically: local ollama, your own GPU server, or on-device MLX — same pipeline, your choice of where gemma thinks.

**one query, full mission**
- type "vessels and trucks at the quay" — the router splits it into tracking targets and a semantic question on its own.
- sam 3.1 tracks every object frame by frame with persistent ids; falcon refines key frames; gemma writes the field report.
- output is the triad ops teams actually need: annotated video, structured json, written report.

**built for the field, not the demo**
- live tab talks straight to field hardware over a 5 fps observer relay — boxes, telemetry, ask-the-scene q&a, and client-side annotated recording.
- line and polygon zone counters, 8-way direction classification, and bytetrack identity come standard.
- fastscan answers "is it there?" in under a minute, then the full pipeline keeps working in the background.
- chunked processing chews through long recordings that would starve a single-pass pipeline.

**an operator's instrument, not a saas dashboard**
- monospace, hairlines, one amber accent. every state is honest: READY, CACHED, MISSING, tracking · frame 240/480.
- three panes: analyze (video missions), inspect (single images), live (field hub). no wizards, no noise.

## Numbers block (real, from a 20 s 480-frame quay clip on this machine)

- 480 frames · 96 processed · 338 unique objects tracked — one query
- ~3.7 fps end-to-end tracking at 512 px, every 5th frame, on apple silicon
- falcon single-image grounding: ~11 s cold, ~0.4 s warm
- model footprint: falcon 9.46 GB · sam 3.1 6.51 GB · gemma 4 e2b 7.2 GB — co-resident on one 32 GB mac
- 0 bytes of footage leaving the machine

## Text bubbles

### bubble set 1 — live ask (operator ↔ scene)
> **operator** — what do you see?
>
> **visionbrain** — holding pattern over the quay. 4 tracked objects in frame, movement nominal. no perimeter events in the last 60 s.

### bubble set 2 — the query, answered
> **operator** — trucks blocking the north access road
>
> **visionbrain** — routing: sam target `truck` · semantic question kept for the report. tracking 480 frames…

### bubble set 3 — field report opening (paper voice)
> field report — port east · berth 12, 14:32
> operation nominal across the surveyed cell. tracked objects held their lanes; no perimeter breaches. recommend continuing the sweep on the current heading.

### bubble set 4 — status chips (small, for layout garnish)
`● all models ready` · `● hub ready · 5 fps relay` · `● recording annotated view` · `complete · report ready`

### bubble set 5 — the promise (for the back page)
> your footage never leaves the building.
> no account. no upload. no per-seat pricing.
> the station is the machine in front of you.

## Where it stands today (honest state of the board)

SHIPPED —
- full pipeline: sam 3.1 track → falcon refine → gemma report, from one query
- ground control web ui: analyze · inspect · live, resizable operator layout
- live field hub: observer relay, detection hud, ask-the-scene, field reports, annotated recording export
- fastscan quick answers; adaptive + chunked processing for long video
- zone counters (line + polygon), 8-way direction tracking, persistent identity
- python api + cli (`visionbrain detect | segment | ocr | track | analyze | fastscan | status`)
- test suite + ci that pass with no models attached

IN PROGRESS —
- fast-path + adaptive sampling hardening (spec'd in FAST_PIPELINE_SPEC.md; quick-answer first, full pipeline behind it)

## Roadmap (direction, not promises)

NEXT —
- alert rules on zones: dwell, loiter, wrong-way, perimeter breach → webhook on event
- multi-feed hub federation: one dashboard, N field relays, per-feed engine control
- thermal + low-light profiles for night patrols

LATER —
- scheduled autonomous sweeps (waypoint missions with per-leg queries)
- report templates per vertical: port ops, terminals, corridors, perimeters
- on-device fine-tune loops: promote operator corrections into few-shot prompts

frame it exactly like this: shipped / in progress / direction. the product doesn't promise what it can't run.

## Vertical one-liners (for business audiences)

- ports & terminals: berth oversight without shipping footage ashore.
- logistics yards: gate-to-gate truck counts, dwell, and lane discipline — one query each.
- corridors & infrastructure: every overpass, every sweep, written up before the drone lands.
- shipyards: basin watch with persistent ids across cranes, barges, and crews.
- security & perimeter: live hub + zones + direction = a fence that writes its own log.

## Call to action (quiet)

- run it: `pip install -e .` → `visionbrain status` → open ground control.
- bring a 20-second clip. leave with a report.
