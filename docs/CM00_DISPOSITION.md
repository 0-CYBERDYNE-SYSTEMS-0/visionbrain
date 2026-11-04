# CM-00 Working-Tree Disposition Record

Date: 2026-09-13

Purpose: working-tree disposition record for CM-00, per
`PRODUCTIZATION_EXECUTION_PLAN.md` — an itemized accounting of every changed and
untracked file in the 2026-09-13 working tree, with a disposition (commit /
ignore / keep untracked) and a proposed commit sequence that slices the
all-in-one WIP snapshot into reviewable, single-concern commits.

Repo owner authorized working-tree disposition in a working session on
2026-09-13; full CM-00 scope decisions (demo workflow, reviewer, budgets,
thresholds) remain open.

## Basis of analysis

- Analysis ref: `wip/snapshot-20260913` = `a365a9c229c7315b58885ab75533e1d834001e54`
  ("WIP: full working-tree snapshot before productization slicing"), diffed
  against base `c226a5d`. All findings below are ref-based; the worktree was
  never diffed.
- The snapshot contains exactly **13 modified tracked files** and **395 added
  (new) files**; no deletions. `.gitignore` is **unchanged** from base.

## Inventory

Modified tracked files (13):

| Path | What changed | Disposition | Target commit |
|---|---|---|---|
| `AGENTS.md` | Adds pointer to NEXT_DEVELOPMENT_SPEC/BUSINESS_CAPABILITY_RECON (ratify CM-00 first); documents `pilot_eval.py` in tree map, per-module test layout, fastscan sampling-honesty and pilot-eval gotchas | COMMIT | 2 |
| `SPEC.md` | Adds `pilot-eval` command section + CLI table row; FrameScore/FrameScores new fields and sampling-honesty algorithm; agent-loop `ChatResponse` tool-call contract; live-engine rewrite (controls `set_threshold`/`set_stream`/`set_task`/`ask`/`report`, clips writer, multi-viewer, fresh-per-pass features, coalesced streams); `/uploads/` auth + job-store eviction in web_app wiring | COMMIT | 2 |
| `src/visionbrain/agent_loop.py` | `VLMClient.chat` now returns a `ChatResponse` (native OpenAI `tool_calls` mapped, malformed JSON args become `arguments_error`/`arguments_raw`); `run_agent` resolves native calls first, textual `<tool>` fallback, retryable user-role error instead of crash; legacy str-returning subclasses still work | COMMIT | 6 |
| `src/visionbrain/cli.py` | Adds `cmd_pilot_eval()` + `pilot-eval` subparser (`--video`, `--ground-truth`, `--report`, `--evidence-dir`, `--tolerance-s`, `--sample-every`, `--max-frames`, `--min-relevance`, `--resolution`) + dispatch entry. **fastscan was NOT touched — it already existed at c226a5d; this diff is pilot-eval wiring only** | COMMIT | 4 |
| `src/visionbrain/frame_selector.py` | New `_select_sample_indices()` spreads sampling across the FULL video (was: earliest-`max_frames` truncation); per-frame Falcon failures recorded as `FrameScore.failed` + `FrameScores.frames_failed`; `sampled_span_s` coverage; negative quick answer states sampled span + failure count; detection totals respect `min_relevance`; single-region summary uses real total (was label string) | COMMIT | 3 |
| `src/visionbrain/live_engine.py` | Major rework: multi-viewer sinks (`_ClientSink`) with one-slot latest-wins coalescing for frames AND detections (empty set is a real value); cross-pass backbone cache removed — fresh features every detect pass, `backbone_every` deprecated with status note; `ask`/`report` controls with one consistent server-side evidence snapshot + shared in-flight slot and `error` wire messages; capture writer daemon thread with single slot spanning post-roll→encode, lazy start, FIFO shutdown; live `set_threshold`/`set_stream`/`set_task`/`set_engine`/`set_vlm` tuning; jpeg_quality/send_width start keys; `_counts_summary` derived server-side | COMMIT | 5 |
| `src/visionbrain/live_tracking.py` | `_ensure_loaded` re-applies `score_threshold` to the cached predictor on cache hits (test doubles tolerated); held re-publish now also holds EMPTY sets so an empty scene can't defeat the `detect_every` throttle | COMMIT | 5 |
| `src/visionbrain/model_host.py` | Checkpoint loads moved OUTSIDE the global lock with per-key `threading.Event` load slots — slow loads no longer stall acquire/release of other keys, concurrent same-key acquires share one load; `_clear_mlx_cache` runs outside the lock on eviction | COMMIT | 5 |
| `src/visionbrain/static/index.html` | Inspect (still) run panel `#inspect-run` with honest job feedback (file/task/phase/elapsed/latest log); live-tab PAUSE + 1 Hz `liveWatchdog` (startup ticks, "feed stalled" overlay, aging stats); threshold slider re-ranged 0.05–0.90 with live re-apply; polygons toggle (`set_task`); JPEG-quality + send-width stream controls (`set_stream`); LFM checkpoint row; localStorage-persisted prompt preset chips (seeded, ≤12) and overlay-layer toggles (MASK/BOX/LABEL/TRACK/CONF); `?live=1` viewer shortcut that attaches to a running hub with saved URL/prompts/threshold; queue-position display; default prompts value removed | COMMIT | 5 |
| `src/visionbrain/supervision_bridge.py` | `weakref.finalize` evicts the per-object compact-mask/shape caches when a `Detections` result is collected — fixes unbounded growth and recycled-`id()` cross-contamination | COMMIT | 5 |
| `src/visionbrain/web_app.py` | `/uploads/{fid}` now requires auth token (401 JSON); `_prune_jobs()` caps the job store at 200 finished jobs (oldest evicted, unfinished never); `_exec` wraps `_exec_run` so launch failures land in `error` state instead of a stuck-`running` job; `serve_upload` uses `_find_upload` (skips sidecar dirs — was serving `{fid}_stills` as a 500) | COMMIT | 5 |
| `tests/test_live_engine.py` | Purely additive (+1072 lines): new classes `TestValidateControlTuning`, `TestWorkerTunables`, `TestBackboneDeprecation`, `TestEvidenceSnapshot`, `TestClipWriter`, `TestDetectionsCoalescing`, `TestAskReportWire` covering the engine rework | COMMIT | 5 |
| `tests/test_visionbrain.py` | **One new class** — `TestSupervisionBridge` (4 tests: compact roundtrip, GC eviction, no cross-contamination, identity without cache entry). Plus added methods in existing classes: `TestCLI.test_pilot_eval_smoke`; `TestWebApp.test_ui_feedback_surfaces_present`, `test_live_simplification_client_pins`; `TestModelHost.test_slow_load_does_not_block_other_keys`, `test_same_key_concurrent_acquire_loads_once`; `TestLiveTracking.test_empty_scene_keeps_detect_every_throttle`, `test_ensure_loaded_reapplies_threshold_on_cache_hit` | COMMIT (mixed concerns — see Risks) | 5 |

Untracked/new files (395, grouped):

| Path | What it is | Disposition | Target commit |
|---|---|---|---|
| `BUSINESS_CAPABILITY_RECON.md` | Root doc: source-based capability + business-foundation recon (2026-09-09/11); reference, not an implementation authorization | COMMIT | 2 |
| `NEXT_DEVELOPMENT_SPEC.md` | Root doc: proposed "Connected Mission v1" implementation contract (Scout + Ground Control one mission); checkboxes deliberately unchecked; ratify CM-00 first | COMMIT | 2 |
| `PRODUCTIZATION_EXECUTION_PLAN.md` | Root doc: proposed productization execution overlay (demo readiness, sequencing, Scout-first paid pilot); requires CM-00 ratification; owns this disposition record | COMMIT | 2 |
| `SIMPLIFICATION_SPEC.md` | Root doc: live-workflow simplification spec (fewer hidden behaviors: choose source, arm prompts, watch detections, ask, save evidence) covering these uncommitted engine/UI/launcher changes | COMMIT | 2 |
| `SUPERVISION_RECON.md` | Root doc: deep recon of roboflow/supervision (releases, 0.26→0.30.2 deltas, ByteTrack deprecation deadline, VisionBrain usage audit) | COMMIT | 2 |
| `docs/architecture-review-20260912-visualbrain.html` | Architecture review brief for VisionBrain ⇄ visionbrain-bridge (Tailwind + mermaid HTML) | COMMIT | 2 |
| `docs/CM00_DISPOSITION.md` | This record | COMMIT | 2 |
| `.gitignore` (append) | Exists in snapshot (26 lines, unchanged from base; `.DS_Store` already covered). Preserve all existing lines; append `.zcode/` | COMMIT | 1 |
| `launch.sh` | Minimal demo launcher: foreground uvicorn Ground Control on :7860 only; prints manual commands for the hub simulator / camera bridge; nothing is ever killed by port | COMMIT | 7 |
| `tools/nyc_cam_fetch.sh` | Demo feed loop: pulls a public NYC traffic-cam JPEG once per second into /tmp for ffmpeg→HLS | COMMIT | 7 |
| `tools/nyc_cam_serve.py` | Quiet threaded static file server for the HLS segments on :8554 (pipe-tolerant) | COMMIT | 7 |
| `marketing/` (~370 files) | Demo + launch assets: `demo/` (captured-app screenshots, annotated MP4s, smart-capture clips, README), `demo60/` (60-second demo video production: script versions, narration WAV/AIFF, cuts, QA frames, final MP4), `shots/` (UI screenshots at various sizes), `social/` (1080x1080/1350/1920 + 1920x1080 card HTML/PNGs + cards.css), `sim/hub.py` (field-hub simulator replaying track JSONs over the observer-relay wire protocol), `showcase.html`, `VisionBrain_Overview.html`+`.pdf`, `copy.md` (marketing copy deck), `runpod-gpu-research.md` (GPU rental cost research) | COMMIT (exclude no `.DS_Store` — none present in the snapshot tree) | 7 |
| `src/visionbrain/pilot_eval.py` | Honest-measurement pilot evaluation harness: `validate_ground_truth()`, `run_pilot_eval()`, `PilotReport` (missed events, frame-level false alerts, latency, coverage, always-present caveats); replay via FastScan scorer with injectable `score_fn`/`frame_reader` | COMMIT | 4 |
| `tests/test_frame_selector.py` | Per-module suite for `frame_selector.py`: `_select_sample_indices` full-span behavior, quick-answer honesty (coverage + failures), dataclass contracts, mocked `score_frames` | COMMIT | 3 |
| `tests/test_pilot_eval.py` | Per-module suite for `pilot_eval.py`: ground-truth validation, detection/miss/false-alert/latency math, coverage + caveats, evidence files, report save/summary — all with injected fakes (CI-safe) | COMMIT | 4 |
| `tests/test_agent_loop.py` | Per-module suite for `agent_loop.py`: native tool-call path, textual `<tool>` fallback, legacy str subclass, malformed-argument retry, `VLMClient.chat` mapping, `_normalize_chat_response` — scripted VLM fakes (CI-safe) | COMMIT | 6 |
| `.zcode/` (1 file: `plans/plan-sess_….md`) | Saved ZCode session plan (notes describing the SUPERVISION_RECON write-up). Local session artifact; no secret values found, but it must never be tracked | IGNORE via `.gitignore`; untrack (see commit 1) | 1 (untrack) |
| `prototype/` (9 files) | Untracked design mockups per AGENTS.md convention: `gallery.html` ("six screens"), `gallery-clay.html`, `gallery-cockpit.html` ("ten cockpits"), `simplification-gallery.html` (3 interface directions), `integration-plan.html` (VisionBrain × supervision), `visionbrain-vs-supervision.html` (recon comparison) + sample JPGs (`group`, `overhead`, `plaza`, `som`) | KEEP UNTRACKED — never commit; untrack (see commit 1) | 1 (untrack) |
| `.DS_Store` | macOS Finder metadata; already ignored by existing `.gitignore` line; none present in the snapshot tree | IGNORE (already covered) | — |

## Dispositions

- **COMMIT** — all 13 modified tracked files.
- **COMMIT** — the five root docs (`BUSINESS_CAPABILITY_RECON.md`,
  `NEXT_DEVELOPMENT_SPEC.md`, `PRODUCTIZATION_EXECUTION_PLAN.md`,
  `SIMPLIFICATION_SPEC.md`, `SUPERVISION_RECON.md`),
  `docs/architecture-review-20260912-visualbrain.html`,
  `docs/CM00_DISPOSITION.md`, `launch.sh`, `tools/`, `marketing/`
  (no `.DS_Store` present under it), `src/visionbrain/pilot_eval.py`, and the
  three new test files (`tests/test_frame_selector.py`,
  `tests/test_pilot_eval.py`, `tests/test_agent_loop.py`).
- **IGNORE via `.gitignore`** — `.zcode/` and `.DS_Store`. The existing
  `.gitignore` (26 lines) is unchanged in the snapshot and **already contains
  `.DS_Store`**; only `.zcode/` needs appending, preserving every existing line.
- **KEEP UNTRACKED (never commit)** — `prototype/` (AGENTS.md convention:
  repo-root design mockups stay out of git) and `.zcode/`.

## Proposed commit sequence

Applied on top of the WIP snapshot branch. Because Alpha's snapshot commit
already added `.zcode/` and `prototype/` to the index, commit 1 must ALSO
untrack them (`git rm -r --cached`) so the KEEP-UNTRACKED disposition actually
takes effect; the commit message below still describes that accurately.

1. **`chore: ignore local session artifacts and OS files`**
   - `.gitignore` (preserve all 26 existing lines, append `.zcode/`; `.DS_Store`
     already present — do not duplicate)
   - Plus `git rm -r --cached .zcode prototype` (untrack the two directories the
     WIP snapshot accidentally committed; files remain on disk, untracked)
2. **`docs: ratify Scout-first productization scope and record working-tree disposition`**
   - `AGENTS.md`, `SPEC.md`, `BUSINESS_CAPABILITY_RECON.md`,
     `NEXT_DEVELOPMENT_SPEC.md`, `PRODUCTIZATION_EXECUTION_PLAN.md`,
     `SIMPLIFICATION_SPEC.md`, `SUPERVISION_RECON.md`,
     `docs/architecture-review-20260912-visualbrain.html`,
     `docs/CM00_DISPOSITION.md`
3. **`feat: make FastScan sampling span the full video and report frame failures honestly`**
   - `src/visionbrain/frame_selector.py`, `tests/test_frame_selector.py`
4. **`feat: add pilot-eval replay harness for honest measurement`**
   - `src/visionbrain/pilot_eval.py`, `src/visionbrain/cli.py`,
     `tests/test_pilot_eval.py`
   - `cli.py` is assigned HERE: its diff is exclusively pilot-eval wiring
     (`cmd_pilot_eval` + subparser + dispatch). It does NOT touch fastscan —
     fastscan already existed at the base commit. No mixed concerns.
5. **`feat: rework live engine for multi-viewer streaming, ask/report evidence, and Ground Control tuning`**
   - `src/visionbrain/live_engine.py`, `src/visionbrain/live_tracking.py`,
     `src/visionbrain/model_host.py`, `src/visionbrain/supervision_bridge.py`,
     `src/visionbrain/web_app.py`, `src/visionbrain/static/index.html`,
     `tests/test_live_engine.py`, `tests/test_visionbrain.py`
   - `tests/test_visionbrain.py` assigned here because the large majority of
     its additions are live-path tests (new `TestSupervisionBridge` class,
     `TestLiveTracking`, `TestModelHost`, `TestWebApp` additions). Its single
     pilot-eval item (`TestCLI.test_pilot_eval_smoke`) technically belongs to
     commit 4 — see Risks.
6. **`feat: add native OpenAI tool-call support with retryable argument errors to agent loop`**
   - `src/visionbrain/agent_loop.py`, `tests/test_agent_loop.py`
7. **`chore: add launch script, field tools, and marketing assets`**
   - `launch.sh`, `tools/` (`nyc_cam_fetch.sh`, `nyc_cam_serve.py`),
     `marketing/` (all ~370 assets; no `.DS_Store` present to exclude)

**Coverage verification:** every path in the inventory above appears in exactly
one commit — the 13 modified files map to commits 2 (AGENTS.md, SPEC.md), 3
(frame_selector.py), 4 (cli.py), 5 (the other nine), and 6 (agent_loop.py);
every COMMIT-marked new file maps to commits 1 (`.gitignore`), 2 (six docs
entries), 3, 4, 5, 6, or 7 (launch.sh, tools/, marketing/). The intentionally
excluded paths (`.zcode/`, `prototype/`, `.DS_Store`) appear in no content
commit; `.zcode/` and `prototype/` are untracked in commit 1. No overlaps, no
orphans.

## Risks and notes

1. **Mixed-concern file: `tests/test_visionbrain.py`.** Its additions span five
   areas: pilot-eval CLI smoke (`TestCLI`), web UI feedback surfaces + live
   client pins (`TestWebApp`), model-host concurrency (`TestModelHost`), the new
   `TestSupervisionBridge` class, and live-tracking throttle/threshold tests.
   It is assigned to commit 5 (majority match). If each commit must carry its
   own passing tests, hunk-split it: the `TestCLI.test_pilot_eval_smoke` hunk
   (@@ -463) goes to commit 4; everything else stays in commit 5. Note the
   repo convention "new CLI commands require a smoke test in TestCLI" is only
   satisfied at commit 4 if that hunk is split forward.
2. **Docs-before-code ordering (commit 2):** AGENTS.md/SPEC.md in commit 2
   describe pilot-eval, the live-engine rework, and native tool calls that land
   in commits 3–6. Within one merged series this is cosmetic; do not merge
   commit 2 alone to main and stop.
3. **`.zcode/` and `prototype/` are already in the WIP snapshot commit.** The
   `git rm -r --cached` in commit 1 untracks them going forward, but they remain
   in the snapshot commit's history. If the WIP branch is ever squashed before
   landing, drop them there instead. `marketing/demo60/shots/sat.txt` and
   `marketing/demo60/shots/raw.rgb` look like scratch render intermediates swept
   up with the demo assets — harmless, but candidates for exclusion if
   marketing/ is re-reviewed.
4. **Agriculture wording:** none found. `marketing/copy.md` was checked for
   cattle/livestock/pasture/farm/herd/crop/agri terms — clean; copy is
   domain-neutral (quay, vessels, trucks, port terminal). `launch.sh`, `tools/`,
   and the root docs are likewise domain-neutral.
5. **Secrets scan:** pattern scan over `marketing/`, `tools/`, `docs/`,
   `prototype/`, `launch.sh`, and the `.zcode/` plan file found no API keys,
   tokens, bearer credentials, or passwords. `tools/nyc_cam_fetch.sh` embeds a
   PUBLIC NYC 311/DOITT camera ID (not a credential). The `.zcode/` plan file
   contains only session-planning prose about writing SUPERVISION_RECON.md —
   flagged as a local session artifact, not as sensitive.
6. **Repo weight:** commit 7 adds ~370 binary-heavy assets (MP4s, WAV/AIFF
   narration, PNGs; e.g. `marketing/demo60/visionbrain_demo_60s.mp4`,
   `marketing/demo/rotterdam_excerpt_20s.mp4`). Fine locally; if this repo is
   ever pushed to a remote, consider Git LFS for `marketing/` media first.
7. **Network note:** `tools/nyc_cam_fetch.sh` and `marketing/sim/hub.py` are
   demo-time utilities; the repo's "no external network calls at runtime" rule
   applies to the `visionbrain` package, which stays clean.
8. **Test safety:** all three new test suites are deterministic and CI-safe
   (scripted fakes / injected `score_fn` / mocked inference; no MLX or weights
   needed). `test_live_engine.py` additions test only pure helpers and faked
   workers, consistent with the no-hardware CI requirement.
