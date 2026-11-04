# Paired baseline manifest — Scout-first productization

Date: 2026-09-13
Purpose: record the identified baselines produced by the working-tree disposition mandated by [PRODUCTIZATION_EXECUTION_PLAN.md](../PRODUCTIZATION_EXECUTION_PLAN.md) (Git and paired-release protocol) and [docs/CM00_DISPOSITION.md](CM00_DISPOSITION.md). Nothing in this manifest authorizes deployment, push, or tagging.

## VisionBrain (`/Users/scrimwiggins/VisionBrain`)

| Item | Value |
|---|---|
| Pre-disposition state | `main` @ `c226a5d81d6018bb631d5225ae461ae83689bc14` + 28 dirty paths (13 modified, 15 untracked) |
| Protective snapshot | branch `wip/snapshot-20260913` @ `a365a9c229c7315b58885ab75533e1d834001e54` (preserved) |
| Disposition record | [CM00_DISPOSITION.md](CM00_DISPOSITION.md) |
| Productization commits | `a74104f` chore: ignore local session artifacts · `2bba024` docs: ratify Scout-first productization scope and record working-tree disposition · `4df76f3` feat: FastScan full-video sampling + honest frame failures · `3f6c794` feat: pilot-eval replay harness · `2c891d9` feat: live engine rework (multi-viewer streaming, ask/report evidence, Ground Control tuning) · `df97a37` feat: agent-loop native OpenAI tool calls · `f5c04d4` chore: launch script, field tools, marketing assets |
| Final baseline | `main` @ `f5c04d475668b3b6553d0c8c7962e8692eae1e96` (= `productization/scout-first-v1`, fast-forward merge) |
| Test evidence | `.venv/bin/python -m pytest tests/ -q`: 317 passed, 4 skipped — identical before and after slicing |
| Excluded from repo by disposition | `prototype/` (untracked by convention, restored on disk), `.zcode/` (ignored, restored on disk), `.DS_Store` |
| Remote state | NOT pushed; `main` is ahead 7 of `origin/main` |

## visionBrain-bridge (`/Users/scrimwiggins/visionBrain-bridge`)

| Item | Value |
|---|---|
| Pre-disposition state | `productization/scout-first-v1` @ `d023b3f` + 1 untracked path (`docs/PB00_DECISION_RECORD.md`) |
| Protective snapshot | branch `wip/snapshot-20260913` @ `e5d177c2f63f1d9cb4f7f022f4ed29a750705322` (preserved) |
| Decision record commit | `1cdeeb6a4116b2d9b6643a53aeb332a4a60ec2ed` "docs: ratify PB-00 Scout-first scope and record paired baseline" |
| Final baseline | branch `productization/scout-first-v1` @ `1cdeeb6` — NOT merged to its `main`, NOT pushed (merge is a cross-repo decision pending companion-PR linkage) |
| Concurrent edits observed 2026-09-13 (quarantined, uncommitted, not ours to reconcile) | `android/scout/.../ScoutActivity.kt`, `android/scout/.../strings.xml`, `bridge/vb_bridge/archive.py`, `bridge/vb_bridge/frame_buffer.py`, `bridge/vb_bridge/server.py`, `tests/test_pb01_repro.py` — appear to be in-flight PB-01 work by another session/teammate |

## Toolchain (recorded at manifest time)

- macOS 27.0 arm64 (Apple Silicon)
- Python 3.14.7 (repo-local `.venv`)
- mlx 0.32.0 · mlx_vlm 0.6.3 (within required `>=0.6.1,<0.6.4` window) · supervision 0.28.0 (within `>=0.28,<0.30` pin)

## Backup bundles (off-repo, verified complete histories)

- `/Users/scrimwiggins/VisionBrain-backups/visionbrain-20260913.bundle` — 479 MB, history through snapshot `a365a9c` (pre-slicing)
- `/Users/scrimwiggins/VisionBrain-backups/visionbrain-20260913-postmerge.bundle` — includes final `main` `f5c04d4`
- `/Users/scrimwiggins/VisionBrain-backups/bridge-20260913.bundle` — 97 MB, history through snapshot `e5d177c`

## Open items pending CM-00 (placeholders, not claims)

- Actual model checkpoint revisions and backend resolution to be recorded at first scoped release.
- Demo workflow, reviewer, exclusions, budgets, and threshold-setting process per VB-P0-01/CM-00.
- Release tag `vb-scout-pilot-v0.1.0` is reserved but NOT created; no checklist or document upgrades capability without its recorded evidence.

## Constraints honored during disposition

No force operations, no push to any remote, no auto-deploy or M2 host changes, no CI-triggered installation. Bridge merge and all pushes remain explicit, authorized, cross-repo-linked steps.
