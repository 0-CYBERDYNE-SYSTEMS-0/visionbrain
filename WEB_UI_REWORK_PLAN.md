# Ground Control — Web UI Rework Plan

**Version 2 · September 2026 · synthesized for handoff**

This document is the single plan for rebuilding the Ground Control web UI
(`src/visionbrain/static/index.html` + the thin HTTP layer in `web_app.py`).
It folds four prior efforts into one decision record and adds the design
calls that tie them together:

| Input | What it contributed |
|---|---|
| `ANDROID_UX_UI.md` | The proven interaction grammar of Bridge (drone) and Scout — engines, validate, prompts, ask/report, chips, honesty rules |
| `design-system/README.md` | The visual language — operator's terminal, mono voice, wheat-amber accent, hairlines, quiet motion |
| `prototype/gallery*.html` | Ten explored cockpit directions; they converge on the **Film · Still · Live** source model and a verb row (**Find / Mask / Track / Count / Read / Ask / Report**) |
| Current `index.html` + `web_app.py` | Working engineering: binary wire-format client, live-engine controls, zone/trigger drawing, MediaRecorder export, SSE job streaming, token auth, upload + job queue |

On conflict: the code's protocol contracts (§12.2) are law; this document
decides everything above them. Nothing in the current UI is protected — a
complete rewrite is authorized — but the *proven* pieces are called out to
keep (§12.1) because they encode hard-won protocol knowledge.

---

## 1. Intent — what Ground Control is

First principles. VisionBrain is a split-brain field vision system: heavy
models on a Mac, thin camera clients everywhere else (drone, phone, and now
the desktop browser). The Android apps nailed the operator loop:

> **pick a source → arm engines → type what to find → execute → watch it
> happen live → ask questions about what's there → keep the evidence**

Ground Control is the **desktop seat of that same loop** — not a dashboard
that watches it. Same simplicity, same real-time execution, plus the things
only a desktop can do well:

- **Volume** — batch film analysis, large exports, tabular data out (the
  "inventory the factory shelves" scenario: film in → detections table →
  CSV out → agent on top).
- **Permanence** — a run history and an export center; every result is a
  first-class artifact, not a transient view.
- **Precision** — keyboard-first operation, zone drawing with a mouse,
  side-by-side artifact inspection.
- **Ingestion** — the desktop is where phone-collected data (sightings,
  snapshots, clips) lands and gets put to work.

The one-sentence test for every screen: *can the operator arm a prompt and
execute it in real time, the same way they would on the phone — and then
take the result with them?*

### 1.1 The unifying interaction model: noun × verb × prompt

The prototype galleries converged on this and it is correct:

```
SOURCE (noun)      ACTION (verb)          TARGET (prompt)
─────────────      ────────────────────   ─────────────────────────────
FILM   · STILL ·   find · mask · track ·  one comma-separated line:
LIVE               count · read ·         "person, forklift, pallet"
                   ask · report
```

Every capability in the system is addressable as one cell of that matrix:

| Verb | FILM (batch video) | STILL (image) | LIVE (real-time) |
|---|---|---|---|
| **find** (detect) | analyze/track mission | detect · sam3 | live detect |
| **mask** (segment) | mission + masks | segment · sam3 | `task=segment` |
| **track** | track mission (IDs, paths) | — | persistent tracks (default) |
| **count** | zone counters over time | — | zones + line-cross/dwell |
| **read** (OCR) | — | ocr | — |
| **ask / report** | report artifact | agent answer | grounded ask/report |
| **fastscan** | relevance triage | — | — |

This is not a new mental model — it is the Android grammar with the
desktop's extra verbs visible. The UI leads with source and verb; the
console below shows only what the chosen cell needs.

---

## 2. Principles (inherited, adapted, one new)

Carried over verbatim from the app family — these are product law:

1. **One screen, nothing buried.** No modal ever blocks the stage. Deep
   settings live in one console, one settings drawer.
2. **The HUD is sacred.** Detections are drawn client-side over a clean
   frame (`OverlayView` semantics on canvas). Chrome may fade or collapse;
   the HUD never does, and chrome never occludes it.
3. **Honest AI.** Ask/Report answer from the post-validate detection list
   only. Nothing armed → the UI says `AI · arm prompts`, it does not
   improvise. Answers carry their grounding (counts, labels).
4. **Latest frame wins** on live sockets — no client queue, ever; drop % is
   a feature to display. Batch jobs queue *visibly* instead (queue position
   is honest too — show it).
5. **Domain-neutral copy.** Operator/site language, no vertical-specific
   wording in defaults or placeholder strings. Presets are user-saved, not
   industry-flavored.

Web-specific additions:

6. **Keyboard-first.** Every primary action has a key (§10). A desktop
   operator should fly without the mouse.
7. **Local-first assets.** No CDN, no Google Fonts in the app shell — the
   product promise is "no cloud, no subscription"; the UI must not phone
   home for glyphs. (Marketing surfaces may keep the CDN; the operator app
   may not.)
8. **Everything is an artifact.** If the pipeline produced it, it has a
   row in the run history and a download. Nothing evaporates on tab switch.

---

## 3. Information architecture

### 3.1 Three sources, one console

Replace `analyze / inspect / live` tabs with three **source modes**:

| Mode | Frame source | Backend | Real-time? |
|---|---|---|---|
| **FILM** | uploaded video file | job queue → `visionbrain analyze / track / fastscan` subprocess | no — batch with live progress |
| **STILL** | uploaded/shared image | `detect / segment / sam3 / ocr / agent` jobs | no — seconds |
| **LIVE** | local engine (file-loop, webcam, RTSP) **or** field hub (`ws://…:8765`) | `WS /api/live/ws` or hub observer relay | yes — the parity surface |

The verb row re-labels per mode (§3.2). One Mission Console on the right
reconfigures its sections per (mode, verb) — exactly how the Scout drawer
reveals extras, generalized.

### 3.2 Verb routing (no separate "mode" dropdowns)

- **FILM verbs:** `find` (mission detect) · `mask` (mission segment) ·
  `track` (track mission w/ IDs + JSON) · `count` (mission + zones preset) ·
  `triage` (fastscan) · `report` (mission + report artifact).
- **STILL verbs:** `find` (auto-routes keyword → falcon/sam3, falls back to
  agent) · `mask` (segment/sam3-segment) · `read` (OCR) · `ask` (agent with
  tools, grounded in detections).
- **LIVE verbs:** one continuous execution — the console *is* the verb
  surface (engines, masks, validate, zones, triggers, watch, ask/report).

Keyword routing (existing `prompt_router`) stays server-side for STILL's
auto task; the verb row is a *pre-selection* the user can always override.

### 3.3 Old → new migration map

| Today | Becomes |
|---|---|
| `tab-analyze` + `#c-mode` (mission/track/fastscan) | FILM + verb row |
| `tab-inspect` + `#c-task` (auto/detect/segment/sam3/ocr) | STILL + verb row |
| `tab-live` mode toggle (local/hub) | LIVE + source picker inside console (engine · hub URL) |
| MISSION SETUP right rail (`#mission-setup`) | Mission Console (right, collapsible to a bottom strip) |
| pipeline stage strip (SAM/Falcon/Gemma orbs) | live status chips + per-run engine badges in the run row |

---

## 4. Layout anatomy

One layout, three states. The stage is the hero; the console is a right
rail (default 400px, drag-resizable, collapsible).

### 4.1 Expanded (default)

```
┌──────────────────────────────────────────────────────────────────────────┐
│ GROUND CONTROL   [FILM][STILL][LIVE]     LINK ●  BRAIN ●  AI ⏸ 3 · 41ms  │ 44px top bar
├───────────────────────────────────────────────────────┬──────────────────┤
│                                                       │ MISSION CONSOLE  │
│                                                       │ ┌ source ──────┐ │
│                                                       │ │ · · · · · · · │ │
│                   THE STAGE                           │ └───────────────┘ │
│        (clean frame + client-drawn detection HUD;     │ prompts ▸ presets │
│         zones, targets, paths draw here too)          │ engines SAM FAL…  │
│                                                       │ masks · conf · val│
│  [answer card docks bottom-left when present]         │ ask / report      │
│  [transport bar bottom-center in FILM results]        │ triggers (live)   │
│  [draw tools bottom-right in LIVE]                    │ export            │
│                                                       │ ── run ────────┐ │
│                                                       │ │   ▶ EXECUTE   │ │
├───────────────────────────────────────────────────────┴──────────────────┤
│ runs: ▸ run 3 · find person · film 04:12 · 812 dets   [video][json][csv] │ 32px runs strip
└──────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Console collapsed (operator wants full stage)

Console slides away; everything live condenses into a 36px bottom strip —
the BRAIN-bar collapsed idiom from the drone app:

```
│ LIVE · SAM+FAL · DET · 4 · 1s · 12fps · 0% drop      [▸ console] ● rec │
```

That summary line is *the* honesty surface: engines · task · object count ·
frame age · fps · drop. Tap to re-expand (`c`).

### 4.3 Empty states

Each mode's empty stage is its drop zone (drag-and-drop + browse + paste
path/URL for film/still; "connect" for live). Copy stays operator-neutral:
`drop footage · mp4 mov avi webm`, `connect a source` — never examples that
name an industry.

### 4.4 Zen mode (`f`)

Console + strips fade (HUD stays — always), pointer-move or any key
reveals. For screening and recording clean annotated output.

---

## 5. The Mission Console — section spec

Sections are collapsible; only relevant ones render per (mode, verb).
Ids are proposed in kebab-case and are **the contract for tests** (§13).

### 5.1 `#console-source`
- FILM: drop/browse → file chip (name · size · duration once probed).
- STILL: drop/browse/paste → thumbnail chip.
- LIVE: segmented `engine | hub`. Engine reveals `file/webcam/rtsp` + res +
  detect-every. Hub reveals URL field (default `ws://127.0.0.1:8765`,
  editable — the Android settings-sheet behavior, not a hardcoded constant)
  + optional token. Connection state feeds the chips (§6).

### 5.2 `#console-prompts` — *the heart, identical semantics to the apps*
- One comma-separated line. **Empty SET = detection off** (sent explicitly,
  so the engine stops; arm states survive).
- Preset chips: `SAVE` stores the current line; tap applies; long-press (or
  `✕`) deletes; persisted in `localStorage` (`vb.presets`). Seed with
  nothing — first-run copy suggests examples in neutral terms (`person,
  vehicle, pallet`).
- `SET` (or Enter) sends. While disconnected in LIVE, the line parks and
  sends on connect (existing `pendingStart` pattern generalizes to a
  `pendingControl` queue flushed on open — control messages only, never
  frames).

### 5.3 `#console-engines`
- Chips `SAM` (default on) · `FAL` · `LFM` — independently armable, all
  consume the prompt line. Wire: `set_engine {sam, falcon, lfm, validate,
  lfm_model}` (hub) / arming maps to engine flags on the local engine.
- `VALIDATE` cycler `off → soft → hard`, label always shows current. Soft
  paints SAM∩FAL agreements white (IoU ≥ 0.40); hard filters overlay **and**
  what ask/report see.
- `lfm_model` select (`lfm | lfm3b`) beside LFM when the hub advertises it.

### 5.4 `#console-render`
- `MASKS` toggle = task (`segment` ⇒ polygons, slower; off ⇒ `detect`
  boxes). Toggling re-sends prompts — never purely visual; label the
  section honestly (`render + task`).
- Layer chips: `MASK · BOX · LABEL · TRACK · SOURCE · CONF` — client-side
  draw filters, persisted (`vb.layers`). Defaults: BOX, LABEL, TRACK, CONF
  on; MASK follows task; SOURCE on when >1 engine armed.
- `CONF ≥` slider 0.05–0.90 (default 0.35 desktop), applies on release.

### 5.5 `#console-model`
- `GEMMA | LFM` segmented, hidden until the hub reports a switchable model;
  selection applies on next ask (`set_vlm {model}`), rejections surface in
  the log, never silently.

### 5.6 `#console-ask`
- Question field (`/` focuses) + `ASK` + `REPORT`. One shared in-flight
  slot — both disable while either runs; a `thinking · gemma` pill pulses
  above the answer card; client timeout 100s with honest timeout copy.
- Answers dock bottom-left as a card (collapsed ≤3 lines, expand capped so
  it never covers the runs strip); `REPORT` prefix marks reports; ✕ or Esc
  dismisses. FILM/STILL reports additionally file themselves as artifacts.

### 5.7 `#console-triggers` *(LIVE only)*
- Draw tools (also mouse-optional via keyboard `z/x/t` + numeric drag, §10):
  `zone-rect`, `zone-line`, `target` (box-prompted ROI tracked as
  `target N`). Counts live in the section header (`3 zones · 2 targets`).
- Trigger row: line-cross toggle · direction select · dwell seconds ·
  clip pre/post seconds. `set_triggers` on apply.
- `WATCH`: condition line + interval + model (`set_watch`). Events and
  clips flow to the events feed (§7.3).

### 5.8 `#console-export`
Contextual per mode — see §8. Presence of the section is constant; its
contents are the honest list of what currently exists to take.

### 5.9 `#console-run`
- LIVE: `▶ EXECUTE` arms/starts (or `■ STOP`). The AI chip is the pause
  surface (pause = empty prompt set, resume restores — prompts survive).
- FILM: `▶ RUN` launches the job (via the verb's endpoint); button becomes
  the live monitor: phase · elapsed · progress · queue position when queued.
- STILL: `▶ RUN` per verb; agent's ask uses its own in-flight slot.

---

## 6. Status & honesty model

Three chips in the top bar, identical state grammars to the apps (they are
learned muscle memory — do not invent new ones):

| Chip | States |
|---|---|
| `LINK` | `● live` green · `connecting…` · `down` red · `— ` idle (FILM/STILL: `engine`/`hub` label instead) |
| `BRAIN` | `READY` · `…` pending · `?` link-down-unknowable · `DOWN · detail` red |
| `AI` | `— ` off · `arm prompts` (nothing armed) · `⏳` waiting · `⏸ paused` (tap = pause/resume) · live `3 · 41ms` (count · frame age) |

Frame age is displayed **always** in live (`· 1s`, `stale 4s`) — detection
lag of 1–4s is a physical truth; legibility is the fix, not denial. The
stats line adds `fps · Mbps · drop%` (all computable client-side from the
binary stream — bytes/frame and frame timestamps already arrive).

Batch honesty: jobs show real subprocess phase + queue position from SSE
heartbeats (already in the backend) — no fabricated progress bars; the bar
that exists tracks *frames reported vs. total*, nothing else.

---

## 7. The results experience

### 7.1 FILM
Progress state (phase · elapsed · frames · latest log line, streaming via
SSE `/api/job/{jid}/stream`) → results state:
- Player with the annotated MP4; HUD-stat strip (frames · objects ·
  detections · fps · length).
- **Detections table** — the desktop superpower: sortable, filterable
  (class · track · confidence), grouped counts view; every row is the
  grounding for ask/report and the CSV export. This table is the
  inventory-management surface: film in → rows out.
- Report panel (Spectral serif, paper card — the one sanctioned serif
  moment, per the design system).

### 7.2 STILL
Annotated image at native resolution, zoomable (wheel · `+ - 0`), with the
detections table for find/mask results, OCR text block for read, and the
answer card for ask.

### 7.3 LIVE
Full-bleed canvas at rAF cadence (§11). Events feed (right of stage or
overlaid list, newest-first, capped): zone crossings with direction,
dwells, watch findings, clip captures (each clip a `▶ clip` anchor to
`/api/clips/<name>`). REC + SNAPSHOT float bottom-right (§8).

---

## 8. Export & data out

Every artifact, one grammar: it exists in the run history with type · size
· created; download or copy-path from there and from `#console-export`.

| Artifact | Source | Notes |
|---|---|---|
| annotated video | FILM mission/track | mp4 from the job |
| detections JSON | FILM track/mission, LIVE session | LIVE export = client-side dump of accumulated items (frames · boxes · labels · scores · track ids · polygons) |
| **detections CSV** | the table | one row per detection — *the inventory export* |
| field report | FILM mission, LIVE report | txt/markdown |
| annotated recording | LIVE | `canvas.captureStream` + MediaRecorder (mp4/avc1, WebM fallback) — overlays included by construction; existing proven code, keep |
| snapshot PNG | any mode | stage canvas → PNG with prompts + timestamp burned into a corner strip; also the "sighting" primitive for future phone-sync |
| clips | smart capture | server-side, served at `/api/clips/{name}` |

### 8.1 The scenario this must nail (acceptance north star)

> Operator drops 40 minutes of factory-floor footage. FILM · find ·
> `pallet, forklift, person`. Watches honest progress. Gets the annotated
> video for review, opens the detections table, filters to `pallet`,
> exports CSV — a complete inventory count with timestamps and track IDs —
> and asks the agent "which aisles ran short?" grounded in those rows.

Every phase gate below is judged against this walk-through.

---

## 9. Design language

Adopt `design-system/README.md` wholesale (it is the brand), with these
application decisions:

- **Dark operator surface.** `--surface-1 #141612`-family, warm inks
  (`--ink-1 #ebe5d2` … `--ink-5 #383428`), hairline rules (`--rule
  #292d25`, `--rule-strong #3a3e34`). Paper (`#f3eee2`) only for report
  cards.
- **Accent — wheat amber `#d4b572`, one focal element per view.** This is
  the call: the Android apps default green `#7DFFA8` (a sunlight/night
  field-legibility choice, and user-tunable there); the desktop flagship
  carries the *brand* accent. Amber = chrome, focus, EXECUTE, live states.
  **Detections never use amber** — they keep the 8-hue track palette
  (`LIVE_PALETTE`, stable per `color_id`), agreements highlight **white**
  (soft-validate). Three color systems, three jobs: amber = you/controls,
  spectral = the world/objects, white = verified truth. Never blend them.
- **Type.** JetBrains Mono everywhere; Spectral for report bodies
  (serif-on-paper only). Self-host WOFF2 in `static/fonts/` with
  `ui-monospace` fallback — **no Google Fonts in the app** (principle 7;
  overrides the design-system's marketing-surface CDN note for this
  surface). 11–13px workhorse sizes, caps labels at `0.16em` tracking.
- **Hairlines not shadows.** The only legal shadow is `inset 0 0 0 1px
  var(--rule)`. Floating chrome over video gets a 1px rule + a translucent
  surface (`rgba(surface, .82)`) — glassy but not glowy.
- **Motion.** 120–280ms `cubic-bezier(.2,0,0,1)`. Console slide, chip
  state, card dock. No scanlines, no pulses except the single sanctioned
  `thinking` pill. Respect `prefers-reduced-motion`.
- **Density.** Desktop-tight but never Bloomberg-claustrophobic: 8px base
  grid, console sections 8/12px padding, chips 24px tall.

Component inventory to build once: status chip · engine chip · layer chip ·
preset chip · slider row · segmented control · section header (collapsible,
chevron rotates, `aria-expanded`) · answer card · run row · artifact button ·
events feed row · draw-tool button. No other chrome shapes.

---

## 10. Interaction map

**Keyboard** (documented in a `?` overlay):

| Key | Action |
|---|---|
| `1 2 3` | FILM / STILL / LIVE |
| `Enter` (prompt line) | SET |
| `/` | focus ask field |
| `space` | FILM: play/pause · LIVE: pause/resume detection |
| `c` | collapse/expand console |
| `f` | zen |
| `s` | snapshot · `r` start/stop recording |
| `z x t` | draw zone-rect / zone-line / target · `Esc` exits draw |
| `[ ]` | prev/next preset |
| `?` | keyboard overlay · `Esc` closes cards/panels top-down |

**Pointer:** drag-drop anywhere on stage; wheel-zoom results; zone/line/target
drawing on the live canvas (client-coords → canvas px → normalized 0–1 —
existing math is correct, keep it); drag console edge to resize (persisted,
as today).

**Back/Escape order:** keyboard overlay → answer card → settings drawer →
draw mode → default.

---

## 11. Performance budget (the parity mandate)

Desktop must meet or beat the phones on the live path:

- **Paint at rAF, not per-message.** WS frames land in a double buffer;
  a `requestAnimationFrame` loop draws latest-only. Never `drawImage` inside
  `onmessage`. Target: 60fps idle draw, zero queued frames.
- **Decode off the hot path.** `createImageBitmap(blob)` (already) is fine;
  never decode the same blob twice; revoke/replace bitmaps to avoid GC churn.
- **Overlay math in one pass.** Build the paint list (boxes/polygons/labels)
  on `detections`, not per-frame; layer toggles must not re-walk the wire
  format.
- **Budgets:** connect-to-first-frame < 500ms (local engine); input-to-paint
  < 16ms/frame at 1080p stage; a 30fps stream must show 30fps (not 12) —
  this is the "same or better than Android" line. Measure with the stats
  line; a regression in fps is a bug, not a tuning note.
- SSE job polling stays 0.08s server-side; the UI must re-render log lines
  in batches (`requestAnimationFrame`-coalesced), not per event.

---

## 12. Engineering plan

### 12.1 Keep (proven, hard-won)

| Piece | Why |
|---|---|
| Binary wire client (`>III` header, jpeg_len @8, pixels @12, telem JSON) | Protocol truth; parity with hub + apps |
| Live control messages + `pendingStart` parking pattern | Extend to `pendingControl` queue |
| Zone/target drawing math + `liveRestoreControls` re-assert order (zones → targets → triggers → watch) | Server accepts during worker spin-up; order matters |
| MediaRecorder export (mp4/avc1 → WebM fallback) | Correct by construction |
| SSE job streaming + queue-position display | Backend already honest |
| Token auth (`withToken`/`authFetch`, 401→prompt→retry) | Deployment reality (DEPLOY.md) |
| `web_app.py` HTTP surface | Zero backend changes required for P0–P1 |

### 12.2 Protocol contracts (do not deviate)

- Hub/engine WS (role `dashboard`): `hello{role}`, `set_prompts{prompts,
  task, threshold}` (hub) — the local engine's `set_prompts` takes
  `{prompts}` only, with `task`/`threshold` set at `start` (extra keys are
  ignored, so one sender is fine), `set_engine{sam, falcon, lfm, validate, lfm_model}`,
  `set_vlm{model}`, `ask{question}`, `report{report_type, summary}`,
  `set_zones`, `add_prompt_box{box,label}`, `remove_targets`,
  `set_triggers`, `set_watch`, `stop`, `shutdown`.
- Inbound: binary frames; `detections{items:[{box, label, score, track_id,
  color_id, direction?, track_state?, polygon?}]}`, `status`, `answer`,
  `ask_ack`, `report_result`, `engine_stopped`, `event`, `capture`.
- HTTP: `/api/upload`, `/api/job/{analyze|track|fastscan|detect|segment|
  ocr|sam3|agent}`, `/api/job/{jid}(/stream|/detections|/report|/fast|
  /file/{kind})`, `/api/clips/{name}`, `/uploads/{fid}`, `/api/status`,
  `/api/settings`, `/api/healthz`.

### 12.3 Front-end structure

Single `index.html` remains the deliverable (repo convention: no build
step), but internally modular: `static/app.css`, `static/app.js`,
`static/fonts/` — served by the existing `/static` mount; `index.html`
references them with absolute paths (`/static/app.css`). Keep zero external
origins. (`index.html` stays the only server-referenced file — `web_app.py`
root handler is unchanged.)

### 12.4 Phases & acceptance

- **P0 — Shell + LIVE parity (the riskiest core).** Layout chrome, console
  (source/prompts/engines/render/model/ask), chips, collapsed strip, rAF
  painter, snapshot, recording. *Accept:* connects to `marketing/sim/hub.py`
  AND local engine (file + webcam); every console control sends its
  documented message; pause/resume, validate cycler, masks task-switch, and
  preset save/apply/delete all work; stats line shows real fps/drop; keyboard
  map complete.
- **P1 — FILM.** Drop → verb-routed job launch, SSE monitor with queue
  position, player + HUD stats, report card, artifact downloads.
  *Accept:* the §8.1 scenario up to (but not including) CSV/agent.
- **P2 — STILL + agent.** Verb routing incl. OCR + sam3; zoomable result;
  answer card; agent job surfaces as ask with grounding.
- **P3 — Detections table + CSV + runs history.** Sortable/filterable
  table, CSV export, run rows with artifacts, LIVE session JSON dump.
  *Accept:* the full §8.1 scenario end-to-end.
- **P4 — Polish & ingestion.** Accent-wheel (sync the Android theming DNA),
  sightings/snapshot import from phone exports, theming persistence,
  a11y audit fixes.

Each phase lands green on §13 before the next starts.

---

## 13. QA gates

1. **Structural test** (add to `TestWebApp`): fetch `/`, assert the
   contract ids exist (`console-source`, `console-prompts`, `console-engines`,
   `console-render`, `console-model`, `console-ask`, `console-run`, chips,
   stage); assert **no external origins** in the HTML (`grep` for
   `https://fonts` / any `http(s)://` asset = fail). Keeps local-first law.
2. **Parity checklist** (manual, per release): every semantic in §4–§6
   behaves identically to the Android table in `ANDROID_UX_UI.md` §4
   (empty-SET-off, pause semantics, validate modes, masks-resend, shared
   ask slot, 100s timeout copy).
3. **Honesty pass:** with nothing armed / link down / stale frames, every
   shown state is one of the documented honest states (§6) — no spinner
   theater, no invented counts.
4. **Performance gate:** stats line ≥ stream fps on the sim hub for 60s;
   zero frame-queue growth; console collapse/expand < 280ms.
5. Existing pytest suite green; `check_ui_inventory`-style discipline
   (that gate itself is bridge-repo-only — here the §13.1 test is our lock).

---

## 14. Known quirks — do not "fix" blindly

- Detection lag (1–4s cadence) and stale auto-hide are *displayed*, not
  hidden. Frame age is the fix.
- No retry queue for live control sends — a dropped send shows a status,
  full stop (latest-frame-wins is the architecture).
- MASKS toggle re-sends prompts; it is a task change, not a render change.
- MediaRecorder mp4 support is Chrome-gated; WebM fallback is correct, and
  the copy says which it saved.
- The local engine is SAM-only today; FAL/LFM/validate chips apply to hub
  connections. Grey them with a tooltip on local-engine LIVE rather than
  hiding (honesty over surprise), until engine support lands server-side.
- `VB_VLM_MODEL` pin hides the model switcher — a deployment feature.
- WebSockets bypass HTTP token middleware; the live URL carries `?token=`
  (existing behavior, documented in DEPLOY.md — preserve).

---

## 15. Open questions (with recommendations)

1. **Hub URL default** — recommend `ws://127.0.0.1:8765` prefilled from
   `localStorage('vb.hub')`, last-used wins, editable in `#console-source`.
2. **Events feed placement** — overlay list right-of-stage (recommended:
   keeps stage geometry stable) vs. docked console section.
3. **CSV columns v1** — recommend `frame_id, ts_ms, label, score, track_id,
   box_x1y1x2y2 (norm), source`; polygon export stays JSON-only.
4. **Multi-hub observation** (watch several field hubs at once) — explicitly
   out of scope until P4 decision; the architecture (role `dashboard`) allows
   it later.
5. **Light mode** — no. The stage is video; dark is the operator surface.
   Paper lives in report cards only.

---

*End of plan. Build P0 first, gate it hard, and everything after falls out
of the same grammar.*
