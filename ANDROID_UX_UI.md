# Android UX/UI Reference — VisionBrain Bridge (drone) & VisionBrain Scout

A complete walkthrough of every screen, control, mode, gesture, and state in the
two Android apps, written so a developer or designer who has never seen the
product can navigate, restyle, or extend either app without breaking anything.

Sources of truth (this doc summarizes them; they win on conflict):

- Drone cockpit layout: `android/app/src/main/res/layout/activity_main.xml`
- Drone cockpit logic: `android/app/.../bridge/MainActivity.kt` (monolithic by design)
- Scout layout: `android/scout/src/main/res/layout/*` (+ `layout-land/*` variants)
- Scout logic: `android/scout/.../scout/ScoutActivity.kt`
- Control-level inventory: `docs/ui-audit/UI_INVENTORY_LOCK.md`, `docs/ui-audit/WIRING_INVENTORY.md`
- User-facing manual: `docs/USER_GUIDE.md`

> **⚠ Gate for the drone app:** `scripts/check_ui_inventory.sh` enforces
> `docs/ui-audit/UI_INVENTORY_LOCK.md`. Every id in `activity_main.xml` is
> locked — you may move/restyle/relabel controls, never delete them or drop
> their Kotlin wiring. Scout has **no** lock gate, but mirror the same
> discipline. Run the gate before declaring any `:app` layout work done.

---

## 1. The two apps at a glance

| | **VisionBrain Bridge** (`:app`) | **VisionBrain Scout** (`:scout`) |
|---|---|---|
| Package | `com.farmfriend.visionbrain.bridge` | `com.farmfriend.visionbrain.scout` |
| Frame source | DJI Mini 4 Pro via MSDK5 (drone camera) | Phone camera (CameraX) **or** a picked/shared still |
| Job | Fly the drone with a full flight HUD while live AI analysis runs | Field tool: point phone at scene / load a photo, get AI boxes + grounded answers |
| DJI dependency | Yes (MSDK5 + vendored UXSDK) | None |
| One screen? | Yes — single cockpit, two modes (FLY / BRAIN) | Yes — single stage, two modes (VIDEO / PHOTO) |
| Theming | User-tunable accent color wheel + brightness | Fixed dark theme (`scout_*` colors) |
| Orientation | One layout, used both ways | Separate portrait/landscape layouts (`layout-land/`) |

Both apps share `:bridge-core` (`BridgeWebSocketClient`, `FrameEncodePipeline`,
`OverlayView`, `StreamForegroundService`), so status chips, engines, prompts,
ask/report, and overlay semantics behave the same way in each — differences are
called out explicitly below.

Shared design language:

- **Dark, glassy, monospace**. Control chrome floats over a full-bleed video
  stage; labels are small monospace caps. Nothing modal blocks the feed.
- **One screen, nothing buried.** Primary actions are always one tap away;
  deep settings live in one sheet (⚙) or one panel (PANEL / CONTEXT).
- **AI HUD is sacred.** Detection boxes/masks/labels are drawn by `OverlayView`
  over the feed and are never hidden by chrome fades, mode switches, or
  immersive idle. Only control chrome fades; vision HUD does not.
- **Honest AI.** Ask/Report are grounded in the detection list only. With no
  prompts armed or nothing detected, the UI says so rather than inventing.
- **Latest frame wins.** Streaming drops backlog by design; UI copy never
  promises a queue.

---

## 2. App 1 — VisionBrain Bridge (drone cockpit)

### 2.1 Screen anatomy

```
┌──────────────────────────────────────────────────────────────┐
│ topScrim:  [DJI TopBar: status·health·mode·gps·rc·hd·batt·⚙DJI] │
│ [CAM][MAP][PANEL][SNAP]            [WS ✓][BRN …][AI ⏳ cow…][⚙APP] │
│ ┌ controlPanel (PANEL dropdown) ┐                             │
│ └───────────────────────────────┘   [settingsSheet 380dp ⚙APP]│
│ [LAUNCH/LAND] [HOME]            ← flightActions rail          │
│                                 flyRail (DJI camera controls)→│
│              (FPV video feed + OverlayView AI boxes fill all) │
│ [TelemetryPanel or MapPanel]                                  │
│ [PFD attitude instrument] [answerCard]   [− 1.0× +][FLY|BRAIN]│
│ ┌─ brainBar (BRAIN mode only) ───────────────────────────────┐│
│ │ BRAIN  [COLLAPSE]  · prompt chips · engines · layers · ask ││
│ └────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────┘
```

Entry points: launcher, plus `UsbAttachActivity` which just relaunches the app
when the RC is attached over USB.

### 2.2 Modes: FLY ⇄ BRAIN

Bottom-right segmented pill `FLY | BRAIN`. The **lit segment is the current
mode**; tap the other segment to switch. Choice persists across launches.

| Region | FLY | BRAIN |
|---|---|---|
| `flyRail` (DJI photo/video/shutter/exposure rail, right edge) | visible | hidden |
| `fpvInteraction` (tap-to-focus on video) | active | disabled |
| `brainBar` (bottom AI console) | hidden | visible |
| Telemetry panel / Map PiP (bottom-left dock) | visible (map wins over telemetry) | hidden |
| PFD attitude instrument | visible | **stays visible** (deliberate) |

Switching modes never stops video, streaming, or AI — detections keep drawing.

### 2.3 Always-on chrome (both modes)

**Top-left button row:**

| Control | Action | Behavior details |
|---|---|---|
| `CAM` | Cycle camera/lens source | Cycles available FPV sources (main ⇄ others). With <2 sources it just flashes the label "Only one camera source". `cameraSourceLabel` shows a transient toast naming the new source. |
| `MAP` | Toggle the map PiP panel | Map and telemetry share the bottom-left dock — opening one closes/hides the other; opening the map dismisses any answer card + thinking pill. |
| `PANEL` | Toggle the shortcut menu (`controlPanel`) — see §2.5 | |
| `SNAP` | Save annotated screenshot | Captures the video frame **with the AI overlay burned in** to the phone's `Pictures/VisionBrain`. Button reads `···` and disables while saving; result comes back as a snackbar. Needs the camera preview up. |
| ⚙ `btnSettings` | Open the **app settings sheet** (§2.6) | This is the *bridge/app* gear — not the DJI gear. |

**Top-right status chips (next to ⚙):**

| Chip | States |
|---|---|
| `WS` (bridge link) | `WS ✓` green = link healthy · `LINK DOWN` red |
| `BRN` (Mac brain) | `BRAIN READY` · `BRAIN …` pending (link up, no status yet) · `BRAIN ?` grey (link down, unknowable) · `BRAIN DOWN · <detail>` red |
| `AI` (detection state, hidden when disarmed) | `AI ⏳ <prompts>` amber = armed/waiting · `AI ⏸ paused` = detection paused · live counts like `SAM 3 · FAL 1 · AGR 1 · 842ms` · **tap the chip to pause/resume detection** without losing prompts |

### 2.4 Flight rail (FLY)

- **LAUNCH / LAND** — DJI `TakeOffWidget`. Tapping opens DJI's own confirm
  dialog (slide to execute). The caption under it flips between `LAUNCH` and
  `LAND` based on live flying state (LAND shown in warning color).
- **HOME** — DJI `ReturnHomeWidget` (RTH), self-contained.
- **flyRail** (right edge, DJI `CameraControlsWidget`) — photo/video mode
  switch, shutter/record, exposure indicator (toggles the exposure panel),
  camera MENU/gear (opens the full DJI settings drawer). RC shutter buttons work.
- **TelemetryPanel** (bottom-left) — altitude, distance from home, H/V speed.
- **PFD** (bottom-center) — horizon/compass/heading instrument; deliberately
  non-clickable and visible in both modes.
- **Zoom rocker** (right, beside mode pill) — `−` / `1.0×` / `+` digital zoom
  via `ZoomController`, 1.0×–4.0× in 0.5 steps. The label shows the live ratio
  from the drone. Zoom changes the same feed the AI sees.

### 2.5 DJI panels & the top bar

`PANEL` (or tapping empty gaps in the top scrim) opens the shortcut menu:

| Row | Destination |
|---|---|
| Aircraft Status | `SystemStatusListPanelWidget` (RTH altitude, max altitude/distance, novice mode, SD, units…) |
| DJI Settings | Full DJI aircraft settings drawer (flight controller, perception, RC, HD transmission, battery, camera, gimbal) |
| Exposure | `ExposureSettingsPanel` (P/A/S/M, ISO, shutter, EV) beside the fly rail |
| Camera Source | Cycles camera source (same as CAM) |
| Gimbal Recenter | Resets gimbal to center. **The RC wheel is the gimbal**; on-screen touch-gimbal is disabled on purpose. |
| Lights/Common | DJI settings → Common tab (LED control; note: the LED toggle itself is unreliable on Mini 4 Pro — known platform limitation) |
| Obstacle/OA | DJI settings → Perception tab |

**Top-bar icons are mostly status lights that also open things** (wired in
`wireDjiCockpit()`): status text → system status list; health pill → warning
list; flight mode → aircraft tab; GPS → UXSDK GPS popover; RC/HD/battery icons
→ their DJI settings tabs; the DJI gear (far right) → full settings drawer.
AirSense and simulator icons are hidden on this build. BACK (or tapping the
button again) closes DJI panels.

### 2.6 Map PiP

Toggled by `MAP`, closed by `✕` (`btnMapClose`), `MAP` again, or BACK. 280×200dp
panel docked above the brain bar. Bundled Leaflet + OSM tiles
(`assets/aircraft_map.html`); tiles need phone internet.

- Aircraft is a **green arrow rotated to true heading**; green trail marks the
  path; dashed blue line points home; blue `H` square marks the home point.
- Status line: `MAP · AC <lat>,<lon>` or `MAP · waiting for GPS`.
- In-map HTML buttons: **FLW** toggles follow mode (auto-center on aircraft);
  **CLR** clears the trail. Double-tap the map also recenters. Pan/zoom gestures
  work when follow is off.

### 2.7 Settings sheet (app ⚙)

Right-side sheet (380dp, full height); cockpit keeps running behind it. Sticky
header with `✕` close. Sections:

- **CONNECTION** — mDNS-discovered Mac servers rendered as tappable rows
  ("VisionBrain Bridge on <host> — ws://ip:port"); tapping a row connects
  (the sheet auto-closes once the link comes up). Below: manual `ws://` URL
  field, optional token field, `CONNECT` / `DISCONNECT` buttons. Defaults to
  `ws://192.168.1.181:8765` (the M2 inference host).
- **STREAM** — `START` / `STOP` streaming; FPS slider 1–30 (default 12);
  JPEG quality 30–95 (default 70); **Binary frames** toggle (protocol v1.1
  binary framing vs JSON+base64 fallback); live stats line
  `fps · Mbps · drop %`; **Speak answers** toggle (TTS, default on).
- **APPEARANCE** — hue color wheel (176dp) + brightness slider. Live-retints
  the whole cockpit (buttons, chips, labels, overlay accent); persists on
  release. Default accent `#7DFFA8` (green).
- **SDK** — DJI registration status (`SDK ✓ / …`) + hint text + `RETRY SDK
  REGISTER` (needs internet once; a DJI *account* login is not required).
- **CONSOLE** — `CONSOLE ▸/▾` toggles a 160dp scrolling monospace log
  (selectable, autoscrolls).

### 2.8 BRAIN bar (BRAIN mode)

Bottom console, height-capped with internal scrolling. Two states:

**Collapsed:** one header strip whose title becomes a live summary —
`BRAIN · SAM+FAL · DET · 4 · 1s` (armed engines · DET/SEG task · object count ·
frame age, or `LINK DOWN` / `WAIT` / `STALE <n>s`), or `BRAIN · OFF · tap
EXPAND` with all engines off. Tap header or `EXPAND` to open. Collapsing never
stops detection — boxes keep drawing.

**Expanded contents (top → bottom):**

1. **Prompt preset chips** (`ChipGroup`) — one saved preset per chip. Tap chip
   = apply it immediately (sets the prompt line and sends). Chip `✕` icon or
   long-press = delete (with **UNDO** snackbar; deleting the chip matching the
   live prompt line also turns detection off). `SAVE` chip saves the current
   prompt line as a new preset. `CLEAR` chip (only shown when presets exist)
   wipes all presets, also with **UNDO**.
2. **Engine strip** — independently armable detectors, all sharing the prompt line:
   - `SAM` (default on) — multi-prompt detect/segment. Green boxes.
   - `FAL` (default off) — Falcon phrase grounding. Amber boxes. Throttled
     ~1 Hz when SAM is also armed.
   - `LFM` (default off) — compact VLM used as a third *detector*, not for ask.
   - `VAL` — cycles validate mode `off → soft → hard` (label shows current):
     soft keeps all boxes and highlights **agreements white** (SAM∩FAL IoU ≥ 0.40,
     compatible labels); hard shows **only** agreed boxes, for overlay and Ask/Report.
3. **Overlay layer chips** — `MASK / BOX / LABEL / TRACK / SOURCE / CONF`,
   toggling each drawn element live (mask fill/outlines, boxes, labels, track
   IDs, detector source tag, confidence %). Choices persist.
4. **VLM strip** — `ASK  GEMMA | LFM` segmented chips routing Ask/Report to the
   Mac's Gemma (sharper, ~11 s swap) or LFM (fast, ~3 s swap). **Hidden until
   the server reports a switchable model**; `VB_VLM_MODEL` pin on the Mac hides
   it permanently. Switch applies on the next ask; rejection surfaces in the log.
5. **Confidence slider** — `CONF ≥ 0.35` readout; range 0.05–0.90. Re-applies
   live on release (no need to re-press SET).
6. **Prompt row** — comma-separated target classes (`cow, fence, trough`),
   **Masks** switch (on = `segment` task with polygons, slower; off = `detect`
   fast boxes), **SET** button. **SET with an empty field turns detection off**
   (engine chips keep their arm state).
7. **Ask row** — question field ("Ask about the scene…"), **🎤** mic, **ASK**.
   - Mic: in-place `SpeechRecognizer` with partial results typed live into the
     field; on final result the question **auto-sends**. While listening the
     button reads `■` on red (tap to cancel). Hidden entirely on devices
     without speech recognition. Enter/SEND in the field also submits.
   - **REPORT** — generates a grounded field report (arrives expanded).
   - ASK and REPORT share one in-flight slot: while one runs, both disable
     (dimmed). A pulsing `Thinking…` pill appears above the answer card.
     Client timeout is 100 s (just past the server's 90 s LLM timeout).

### 2.9 Answer card & speech

Compact card docked bottom-left (~55% width) above the brain bar, clamped to
**3 lines** collapsed. Header shows `Q: <question> ▾ more / ▴ less`; tapping
the card or header expands it (capped so it never covers the LAUNCH/HOME
rail). Long-press the body or tap `✕` to dismiss (dismiss also stops speech).
Successful answers/reports are spoken via TTS when **Speak answers** is on —
stripped of markdown before speaking.

### 2.10 Gestures & system behavior (drone app)

- **Tap video (FLY only)** — tap-to-focus / spot metering with sun-slider EV
  (DJI `fpvInteraction`). Disabled in BRAIN so taps can't fight the AI overlay.
- **BACK stack order** — DJI panels → map PiP → answer card → settings sheet →
  default (app exit).
- **RC hardware** — sticks always fly; RC wheel = gimbal; RC shutter/record
  buttons drive the camera rail.

---

## 3. App 2 — VisionBrain Scout

### 3.1 Screen anatomy

```
┌────────────────────────────────────────────┐
│ topChrome (row 1): [LINK][BRAIN][AI] chips │
│ (row 2): [VIDEO][PHOTO][CONTEXT][📷][JOURNAL]│
│                                            │
│        full-bleed stage: camera/still      │
│        + OverlayView AI boxes (centerCrop) │
│                                            │
│ [statsLine ·· fps/Mbps/drop]               │
│ [statusLine — session messages]            │
│ [answerCard — BRAIN RESPONSE]              │
│ ┌ brainStrip (bottom drawer) ┐             │
│ │ ── handle ──               │             │
│ │ prompts · presets · engines│             │
│ │ question · ASK · REPORT    │             │
│ │ (MORE… → threshold/model/  │             │
│ │  overlay chips)            │             │
│ └────────────────────────────┘             │
│   (panelScrim + CONTEXT panel / JOURNAL    │
│    panel / snapshot preview slide over)    │
└────────────────────────────────────────────┘
```

Entry points: launcher, or **sharing an image from any app** (opens straight
into PHOTO mode with that image).

### 3.2 Immersive behavior (the defining Scout interaction)

- The app runs full-screen immersive; control chrome **auto-fades after idle**
  (`ImmersiveIdlePolicy`: waits while a panel is open, keyboard is up, or the
  drawer is expanded — then collapses the drawer first, then fades all chrome).
- **Tap anywhere on the open stage** toggles control chrome (reveal/hide).
  Vision HUD (boxes/masks/labels) is *never* hidden.
- Any touch on visible chrome re-arms the idle timer. Alert-type status
  messages (fail/denied/required/…) re-reveal the chrome and announce for
  accessibility.

### 3.3 Top chrome

Row 1 — status chips (horizontally scrollable):

| Chip | States |
|---|---|
| `LINK` | `LINK —` initial · `LINK LIVE` green · `LINK …` connecting · `LINK FAIL` red · `LINK DOWN` |
| `BRAIN` | `BRAIN —` · `BRAIN READY` · `BRAIN · <detail>` (e.g. waiting frames) · `BRAIN DOWN · <reason>` red / `BRAIN DOWN · no --brain` |
| `AI` (tappable) | `AI —` · `AI · arm prompts` · `AI ⏳ resuming` · `AI ⏸ paused` · `AI CLEAR · <ms>` · live `AI <n> · <top labels> · <ms>` — **tap to pause/resume detection** (refuses with nothing armed) |

Row 2 — actions:

| Control | Action |
|---|---|
| `VIDEO` toggle | Camera mode (default). Requests camera permission; if denied, status says "PHOTO mode remains available". Starts the shared foreground service while streaming. |
| `PHOTO` toggle | Opens the system image picker; loads the still (EXIF-corrected, downscaled), stops the camera, and starts the still-resend loop (`PhotoResendController` resends the JPEG until the Mac acknowledges that frame id, so photo detections are guaranteed to arrive). Canceling the picker keeps the current mode. |
| `CONTEXT` | Opens the full-screen Context panel (§3.6). |
| 📷 `btnSnapshot` | Captures an **annotated snapshot** (frame + overlay + prompts + GPS) into the sightings journal. Haptic confirm + "Sighting saved" toast. Disabled until a frame exists. In PHOTO mode it prefers the exact frame the detections were computed on (provenance match). |
| `JOURNAL` | Opens the Sightings journal (§3.7). |

Landscape: both rows merge into one 56dp horizontally-scrolling row (same ids,
same behavior).

### 3.4 Brain strip (bottom drawer)

Gel strip docked at the bottom with a drag-handle-looking header (`──`).
Tap the handle to expand/collapse the **extras drawer**. Portrait heights:
collapsed ≤220dp, expanded ≤55% of screen; landscape uses fixed 140dp/82%.

Always visible (expanded or not):

- **Perception prompts** field (`cow, fence, vehicle…`) + **SET** (keyboard
  Done key also applies). Prompts are comma-separated and mirrored into the
  Context panel's field. First install is seeded with `person, face` so the
  first connect produces boxes; blank values are re-seeded (a known
  "connected but no AI" failure mode).
- **Built-in preset buttons**: `LIVESTOCK` → `cow, sheep, horse` ·
  `INFRASTRUCTURE` → `fence, gate, trough, vehicle` · `WILDLIFE` →
  `deer, coyote, dog`.
- **Custom presets**: `SAVE` stores the current prompt line as a one-tap
  preset button (rendered after the built-ins, both here and in Context).
  Long-press a saved preset to delete it.
- **Engine toggles**: `SAM` (default on) / `FAL` / `LFM` — same semantics as
  the drone app; persisted independently.
- **VALIDATE toggle** — label always shows the current mode
  (`VALIDATE OFF/SOFT/HARD`), tap cycles `off → soft → hard`; checked = not
  off. Default **soft**.
- **Question** field ("What do you see?") + **🎤** — Scout's mic launches the
  **system speech dialog** and fills the field with the result (no auto-send;
  the user taps ASK). Unavailable → toast "Voice input is unavailable".
- **ASK** / **REPORT** buttons — disabled until connected; both dim + disable
  while a model call is in flight (shared slot). Keyboard Send in the question
  field submits. Timeout at 100 s surfaces "No response after 100s…".

**MORE…** (or the drawer handle) reveals the extras section:

- **Detection threshold** slider — 0.05–0.95, default **0.15**, live label;
  applies on release.
- **MODEL** row — `GEMMA | LFM` (hidden until the server reports a switchable
  model; `isSelected` highlights the active one; rejected switches show a status
  message).
- **OVERLAY** chips — `MASK / BOX / LABEL / TRACK / SOURCE / CONF` toggles,
  kept in lockstep with the Context panel's copies. Scout defaults: BOX, LABEL,
  TRACK, CONF **on**; MASK **off** (MASK on = segment task — slower polygons;
  off = fast detect boxes; toggling MASK re-sends prompts); SOURCE off.
- `MORE…` also opens the full Context panel.

### 3.5 Answer card

"BRAIN RESPONSE" card slides up between the stage and the brain strip (full
width, scrollable, selectable text). Answers arrive here; **FIELD REPORT**
prefix marks reports; failures show `⚠ <reason>`. **Scout speaks every
successful answer/report via TTS** (no user-facing toggle — unlike the drone
app's "Speak answers" switch). Close via ✕ or BACK.

### 3.6 Context panel (full-screen)

Slides in from the right over a scrim (tap scrim or ✕ to close; discovery
stops on close). Collapsible sections — each header (`PROMPTS ▾` etc.) folds
its section:

- **PROMPTS** — the built-in + custom preset row and prompt field (mirrors the
  brain strip; editing either updates both).
- **ENGINES** — SAM/FAL/LFM toggles + VALIDATE cycler + threshold slider
  (mirrored with the drawer).
- **OVERLAY** — the six layer toggles (mirrored).
- **MODEL** — GEMMA/LFM (hidden until server reports switchable models).
- **STREAM** — FPS slider 1–24 (default **8**, "8 fps keeps the phone light"),
  JPEG quality 30–100 (default **60**), **Binary frames** switch (default on,
  JSON+base64 fallback off).
- **LINK** — server URL field (`ws://host:8765`, prefilled with last used or
  the M2 default), **DISCOVERED SERVERS** — mDNS results appear as rows while
  the panel is open; tapping a row **fills the URL field** (does not auto-connect,
  unlike the drone app). Token field. `CONNECT` (keyboard Go/Done also connects) /
  `DISCONNECT` (enabled only while connected).

### 3.7 Sightings journal

Full-screen panel over the scrim:

- Header: `SIGHTINGS` title, **MAP/LIST** toggle button (label shows the view
  it switches *to*), ✕ close.
- **List** — newest-first entries; each row shows UTC timestamp, detected-label
  summary, and location. Empty state: "No sightings yet. Capture a snapshot to
  add one."
- **Map** — bundled Leaflet map (`assets/sighting_map.html`) plotting geotagged
  sightings. Only entries with GPS plot. Reopening the journal always starts on
  the list; the map WebView stays warm.
- **Tap an entry** → full-screen **snapshot preview**: the annotated PNG +
  caption (time · location · labels). ✕ or BACK closes back to the journal.

### 3.8 Gestures, permissions, back stack (Scout)

- **Stage tap** toggles chrome (§3.2). No pinch/tap-to-focus — the stage is a
  decoded-frame ImageView, not a camera preview surface.
- **Permissions** requested together: camera (VIDEO mode), fine+coarse location
  (geotagging snapshots), post-notifications (33+; the streaming foreground
  service). Location/camera never gate the stream itself.
- **BACK order**: snapshot preview → context/journal panel → answer card →
  expanded drawer (collapses) → default (exit).
- **Sharing**: share an image into Scout → PHOTO mode with that image
  (survives rotation via saved URI).

---

## 4. Cross-app concepts (identical semantics)

| Concept | Meaning in both UIs |
|---|---|
| Engines | SAM (segment/detect) · FAL (Falcon phrase grounding, amber boxes) · LFM (VLM-as-detector). Independently armable; all consume the one prompt line; arm state sent live via `set_engine`, overriding the Mac's startup flags. |
| Validate | `off → soft → hard`. Soft highlights SAM∩FAL agreements white (IoU ≥ 0.40); hard shows only agreements. |
| MASK / task | MASK on ⇒ `task=segment` (polygon masks, slower). Off ⇒ `task=detect` (fast boxes). |
| Prompts | Comma-separated class names. **Empty SET = detection off** (both apps send the empty set explicitly so the server stops). |
| Presets | Saved prompt lines. Built-ins only in Scout (farm-flavored); SAVE/tap-apply/long-press-delete in both; the drone app adds CLEAR-all + UNDO snackbars. |
| Ask / Report | Grounded in the post-validate detection list only — no free-form scene invention. One in-flight slot shared by both. Answers spoken via TTS. |
| Model switch | GEMMA (default ask model) vs LFM (fast/light). UI hidden until the Mac advertises switchability; switching is lazy + evicting on the Mac (~3 s LFM / ~11 s Gemma). |
| Detection pause | Tap the AI chip. Sends an empty prompt set (resume restores it); prompts survive. Photo-mode acks in Scout are never gated by pause. |
| Latest-frame-only | Neither app queues frames; stats lines show live `fps · Mbps · drop %`. |

## 5. Drone vs Scout cheat sheet

| | Drone (`:app`) | Scout |
|---|---|---|
| Modes | FLY / BRAIN (tool rails swap) | VIDEO / PHOTO (frame source swaps) |
| Settings home | ⚙ right-side sheet (CONNECTION/STREAM/APPEARANCE/SDK/CONSOLE) | CONTEXT full-screen panel (PROMPTS/ENGINES/OVERLAY/MODEL/STREAM/LINK) |
| Server discovery tap | connects immediately (sheet auto-closes) | fills the URL field |
| Accent theming | user color wheel + brightness, persisted | fixed palette |
| Snapshot destination | `Pictures/VisionBrain` (one-off PNG) | journal (`ScoutEvidenceStore`) + map |
| Speak answers | toggle in ⚙ (default on) | always on |
| Mic | in-app recognizer, partials live, **auto-sends** | system dialog, fills field, manual ASK |
| FPS / JPEG defaults | 12 / 70 (range 1–30 / 30–95) | 8 / 60 (range 1–24 / 30–100) |
| Threshold default | 0.35 (range 0.05–0.90) | 0.15 (range 0.05–0.95) |
| Chrome fade | none (cockpit always visible) | immersive auto-fade + stage-tap toggle |
| Map | dedicated PiP (aircraft arrow, trail, FLW/CLR) | journal sightings map only |
| Orientation layouts | single shared layout | separate `layout-land/` variants |

## 6. Known quirks a designer should not "fix" blindly

- **Two gears**: app ⚙ (top-right) = bridge/stream settings; the far-right gear
  in the DJI top bar = aircraft settings. Renaming one without the other
  breaks the mental model documented in USER_GUIDE.
- **Top scrim "tap empty area"** opens PANEL, but top-bar children consume
  touches first — only genuine gaps fire.
- **Lights row** lands on DJI's Common tab because the direct LED toggle is
  unreliable on Mini 4 Pro (platform limitation, not a wiring bug).
- **Scout MASK toggle re-sends prompts** (task change requires it) — toggling
  MASK is not a purely visual change.
- **No send queue**: every send returns false when the link is down and the UI
  drops the action with a status message; there is deliberately no retry queue.
- **Mic button disappears** (drone app) on devices without speech recognition —
  that's the feature degrading, not a layout bug.
- **Detections lag video by 1–4 s** (SAM cadence); boxes older than 3 s
  auto-hide. The collapsed brain header shows frame age precisely so lag is
  legible.

## 7. Accessibility notes (both apps)

- Status chips use `accessibilityLiveRegion="polite"`; Scout alert statuses
  call `announceForAccessibility`.
- Scout manages focus explicitly: panels move focus to their close button on
  open and return it on close; underlying chrome is marked
  `IMPORTANT_FOR_ACCESSIBILITY_NO_HIDE_DESCENDANTS` while a modal panel is up;
  each panel sets an accessibility pane title.
- Collapsible sections expose Expanded/Collapsed state descriptions; sliders
  expose value state descriptions (threshold, fps, quality).
- Custom preset buttons carry content descriptions ("tap to apply, long-press
  to remove"); mic/listen states swap content descriptions, not just icons.
