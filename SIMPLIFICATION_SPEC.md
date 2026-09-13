# VisionBrain live workflow simplification

Status: implementation specification; no implementation authorized by this document alone.
Date: 2026-09-06

## Goal

Make the existing operator loop reliable with fewer hidden behaviors: choose a
source, arm prompts, watch current detections, ask questions, and save evidence.
Prefer deleting unnecessary behavior before introducing new machinery.

This specification covers the current uncommitted live-engine, UI, and launcher
changes. It supplements `SPEC.md`, the existing API reference, and does not
replace `WEB_UI_REWORK_PLAN.md`. The full Film/Still/Live redesign is out of scope.

## Decisions and boundaries

- Keep upload authentication, compact-mask cache cleanup, empty-scene detection
  throttling, job failure handling, bounded finished-job history, and visible
  waiting/stalled states. They address concrete problems.
- Keep existing multi-viewer support. The investigation did not establish that
  it is unnecessary; deleting it would remove working behavior. Do not add
  viewer roles, ownership arbitration, or a subscription framework here.
- Keep asynchronous clip encoding, prompt presets, and existing stream tuning.
  Their removal is not justified by code size alone.
- Keep local and external-hub protocol differences explicit. Do not change the
  field hub or upstream model repositories.
- Do not refactor the model residency host or canonical `LiveSamTracker` as
  part of this work. Preserve unrelated work and all marketing/prototype assets.
- No new dependencies, automatic retries, background service manager, or new
  settings framework.

## 1. Remove demo coupling from normal startup

`launch.sh` currently starts a public-camera fetch/transcode/server pipeline and
kills processes by port. Neither is necessary to launch Ground Control.

Normal startup must launch only the web application using the repository venv.
Remove automatic camera-bridge startup, broad port-based termination, and the
corresponding catch-all stop behavior. An occupied application port should
produce the ordinary server bind error; it must not cause another process to
be killed. Ctrl-C stops the foreground application.

Keep demo scripts and assets available for explicit manual use. Do not build a
replacement orchestration layer. Update launcher help to describe its actual
scope; unsupported legacy arguments must fail clearly rather than be ignored.

Remove the UI's hardcoded HLS source and fallback prompts from `?live` startup.
The shortcut may open the Live tab and connect as a viewer, but must not send
`start` automatically. A remembered source may populate the form; starting an
idle engine requires the existing Start action.

Acceptance: opening the shortcut with empty browser storage starts no source;
normal launch starts no fetcher, transcoder, simulator, or camera server; an
occupied port remains owned by its original process.

## 2. Give controls one supported meaning

In local mode, show SAM as the supported detection engine and hide unsupported
Falcon/LFM detection switches. Preserve VLM selection for Ask/Report. Keep hub
engine controls and their wire messages unchanged. Retain the local
`set_engine` compatibility response for existing clients.

Keep the distinction between generating mask polygons and showing overlays,
but label it explicitly: the task control requests masks; overlay controls
change rendering only. When boxes are enabled, draw available boxes even if an
item also contains a polygon and the mask overlay is hidden.

For local `set_task`, remove the extra `set_prompts` send. The local backend
does not clear prompts on task change. Preserve the external hub's documented
task-change prompt behavior. A local mask toggle while paused must leave
detection paused and must not overwrite the saved resume prompts.

Acceptance: mode switching exposes only supported engine choices; local mask
changes send one task message; pause survives those changes; box visibility
works for polygon-bearing detections.

## 3. Remove speculative local backbone reuse

The local WebSocket engine defaults to reusing image features for three
detection passes. This weakens the relationship between the displayed frame
and a supposedly new observation. No accuracy/latency benchmark was performed
during this review to justify that default.

Recompute image features on every scheduled detection pass. Remove the local
worker's cross-pass backbone/encoder cache state and its separate cache cadence.
Retain `detect_every` as the existing inference throttle. Cache use within a
single pass may remain where required by the model helper.

For compatibility, accept the existing `backbone_every` start field but do not
use it to skip fresh features; report a concise deprecation status when it is
provided. Document this behavior in `SPEC.md`. Do not change the independent
canonical tracker API.

Tradeoff: inference can become slower. Accept that cost for trustworthy fresh
observations; any later cache optimization needs measured detection quality,
latency, and accurate observation provenance. Remove unsupported speed claims.

Acceptance: each scheduled local detection consumes that pass's image; skipped
passes do no inference; empty prompts still do no inference; file looping and
pause/resume cannot reuse features from an earlier scene.

## 4. Bound clip work and stop its writer

The writer currently starts in the worker constructor, waits on an unbounded
queue, and has no caller that sends its documented shutdown sentinel.

Start the writer lazily when the first completed capture needs encoding. Allow
one capture across collection, queued work, and encoding. While occupied, later
triggers still emit their events but do not allocate another capture. This
preserves the intended one-capture-at-a-time behavior without a growing queue.

Every worker exit path must signal writer shutdown. Allow an already completed
capture to finish; discard an incomplete post-roll capture on stop. Do not block
the event loop waiting for encoding. The writer must exit after its accepted
work; do not add unsafe thread cancellation or claim to recover a hung codec.

Acceptance: constructing a worker creates no writer thread; repeated start/stop
with a mocked successful encoder returns writer count to baseline; trigger
bursts remain bounded; encoding failures release the capture slot; encoding
never executes on the frame-processing thread.

## 5. Keep only current replaceable stream state

Keep per-viewer JPEG coalescing. Apply the same latest-value behavior to
`detections`, including empty sets, so a slow viewer cannot accumulate old
detection snapshots behind current video. Preserve event, capture, status, and
request-result messages in order. Do not introduce a generic message broker or
claim that this bounds every possible source of outbound traffic.

Acceptance: a burst of frames and detections retains only the latest pending
value of each per viewer; an empty detection update supersedes older objects;
events are preserved; a slow viewer does not delay another viewer. Existing
last-viewer disconnect and explicit stop behavior remain intact.

## 6. Make Ask/Report completion and evidence consistent

Retain one shared backend inference slot. A 100-second UI timeout means the
answer is late, not that backend inference was cancelled. Keep both buttons
disabled while that request remains pending, show elapsed/late status, and
accept a late result. Remove the current state where buttons appear enabled
but click handlers silently refuse them.

On local rejection or failure, send the existing structured `error` message
shape and clear the request UI. Success and disconnect also clear it. Do not
parse human-readable status text to infer request failure. Preserve external
hub message compatibility. No automatic retry or new cancellation endpoint.

Capture frame, detection records, and active prompts as one consistent
server-owned evidence snapshot before dispatching Ask/Report. Derive report
counts from those records rather than trusting the browser's summary. Preserve
the inbound summary field for compatibility, but do not use it as authoritative
detection evidence. Clear stored detection evidence on pause so earlier counts
cannot survive as current observations.

When prompts are unarmed or a current observation is unavailable, return an
explicit unavailable-evidence response without VLM inference. An observed empty
set is distinct from having no observation. Keep this grounding rule scoped to
local live requests; do not rewrite shared VLM behavior for other workflows.

Acceptance: rejection, inference failure, disconnect, normal success, and late
success leave truthful button state; concurrent requests cannot start a second
inference; a forged browser summary cannot change reported counts; pause clears
old evidence; captured frame and records belong to the same observation.

## Implementation and verification order

1. Remove launcher/shortcut automation and redundant local control behavior.
2. Remove local cross-pass feature reuse; verify fresh observation semantics.
3. Bound and close clip work; coalesce replaceable detection updates.
4. Correct request state and evidence snapshots.
5. Update affected API descriptions in `SPEC.md`, including deprecated cache
   tuning, stream replacement behavior, capture lifecycle, and live evidence.

Add focused behavioral regressions to `tests/test_live_engine.py` and relevant
existing test classes. Use mocked model/encoder calls; tests must work without
weights or MLX hardware. Clean up all test loops and threads. Verify browser
controls and timeout behavior with mocked sockets/timers, followed by a manual
local/hub smoke check where those backends are available.

Run `.venv/bin/python -m pytest tests/ -v` after focused checks pass. Record any
unavailable hardware checks explicitly. Do not infer real-time performance from
mocked tests. Completion requires the acceptance checks above, an unchanged
external-hub wire contract, and no unrelated edits.
