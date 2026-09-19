# Next development spec: Connected Mission v1

Date: 2026-09-11. Status: implementation contract for this cycle, ratified by CM-00
([docs/CM00_DECISION_RECORD.md](docs/CM00_DECISION_RECORD.md), 2026-09-15).
Implementation checkboxes are deliberately unchecked until their recorded evidence
exists. This document does not
authorize deployment, production writes, hardware purchases, or aircraft motion.

## 1. Deliverable and scope

Build one private, single-source installation where Scout and Ground Control
operate the same durable mission. The mission can investigate an uncertainty,
request better evidence, and perform one approved, verifiable sandbox business
action. Keep existing individual analysis and capture tools usable.

This is one product with reusable modules. Retaining two repositories for this
milestone is an incremental delivery choice, not a permanent ban on combining
them. Repository layout must not determine customer-facing fragmentation.

The proposed acceptance fixture is an asset/order checklist: identify three
visible labeled items, resolve one ambiguous label through a requested close-up,
and update a sandbox checklist after review. It exercises the warehouse idea
without declaring warehouse software the chosen market. Item IDs and quantities
are test data, not claims about current visual accuracy.

In scope: shared mission identity/state, scoped controls, durable evidence,
bounded reasoning, human-requested close-ups, sandbox reconciliation/action,
operator review, restart/reconnect behavior, and measured acceptance.

Out of scope: simultaneous cameras; cross-camera re-identification; automatic
flight/robot motion; rescue readiness; voice/wake-word control; real inventory
writes; customer outreach; RF-DETR integration; training pipelines; cloud/VPS
migration; multi-tenant SaaS; a full UI rewrite. These are later increments, not
capabilities delivered by this milestone.

## 2. Read first and reconcile existing work

- [AGENTS.md](AGENTS.md): operating instructions and dependency constraints.
- [Business reference](BUSINESS_CAPABILITY_RECON.md): platform intent, mission
  profiles, and company/runtime fleet distinctions.
- [Bridge handoff](../visionBrain-bridge/NEXT_DEVELOPMENT_SPEC.md): field-owned
  tickets and protocol implementation surfaces. This file owns shared semantics.
- [Adaptive controller detail](../visionBrain-bridge/ADAPTIVE_MISSION_CONTROLLER_SPEC.md):
  expertise input, Inspect/Watch profiles, typed perception tools, live pacing,
  execution-generation rules, and model qualification for CM-02/05/06/08.
  Reuse that detail rather than design a second controller. Its perception-only
  profile is an earlier slice; completing it does not close the business gates.
- [SPEC.md](SPEC.md): existing module interfaces; update it with implemented
  changes rather than treating this proposed spec as proof they already exist.
- [SIMPLIFICATION_SPEC.md](SIMPLIFICATION_SPEC.md): existing live correctness work.
- [WEB_UI_REWORK_PLAN.md](WEB_UI_REWORK_PLAN.md): presentation/interaction target;
  add mission behavior without creating a competing UI redesign.
- [FAST_PIPELINE_SPEC.md](FAST_PIPELINE_SPEC.md): separate in-progress performance
  work. Its concurrency/cache assumptions require source and hardware validation.

The current checkout contains other developers' uncommitted work. Reconcile it
before assigning a ticket; preserve it and reference the eventual paired commits.
This spec was prepared from source inspection, not a new test run or M2 trial.

| Earlier gap | Source now present | What remains before closing it |
|---|---|---|
| Native tool calls discarded | `ChatResponse` and native-call normalization in [agent_loop.py](src/visionbrain/agent_loop.py); [tests](tests/test_agent_loop.py) | Review and run regressions; test a complete conversation with the configured backend. Current loop chooses the first call: define/reject unsupported multi-call turns explicitly. |
| FastScan count, opening-only sampling, hidden failures | Distributed sample selection, summed detection counts, failure/coverage fields in [frame_selector.py](src/visionbrain/frame_selector.py); [tests](tests/test_frame_selector.py) | Run regressions and verify serialized/UI semantics. Summed detections are not unique inventory; distributed sampling is not exhaustive absence proof. |
| Binary timestamp overflow | Low-32-bit masking in [bridge protocol](../visionBrain-bridge/bridge/vb_bridge/protocol.py); updated protocol tests | Run Python/Kotlin regressions; define full-time provenance separately from the wrapped header. |
| No task evaluation harness | [pilot_eval.py](src/visionbrain/pilot_eval.py), CLI wiring, and [tests](tests/test_pilot_eval.py) | Review/reuse it. Extend for mission/action outcomes; its media-timeline detection offset is not camera-to-answer wall-clock latency. |

Do not implement these fixes again simply because the earlier reconnaissance
listed them. Source-present, test-passing, hardware-verified, and deployed are
separate statuses.

## 3. Proposed runtime decisions

The lead ratifies these defaults in ticket CM-00 before parallel implementation.
If a default changes, revise this shared contract and the bridge handoff first.

1. **One runtime authority.** In integrated mode, host a reusable
   `visionbrain` mission runtime inside the existing bridge process. The bridge
   owns capture/transport; the imported runtime owns mission state and decisions.
   Ground Control's browser and Scout connect to that hub. The Ground Control
   HTTP process does not create a second mission database/controller.
2. **Reusable core.** Keep the runtime importable without MLX or bridge imports.
   Inject perception, source-control, storage, and business adapters. Reuse
   existing model wrappers; avoid another detector implementation or agent
   framework. The deterministic replay adapter and field adapter exercise the
   same runtime interface.
3. **One active producer.** A mission binds one approved source connection at a
   time. Reject a competing producer instead of letting it overwrite the shared
   latest-frame buffer. Reconnect or source replacement requires a new source
   epoch and invalidates old current-observation state. Store multiple missions,
   but allow only one executing or waiting-on-human mission to own the source;
   pause/cancel it explicitly before another mission acquires control.
4. **One inference admission authority per host.** Integrated model calls share
   a serialized executor/admission gate; the existing process-local caches and
   bridge GPU lock are not a machine-wide scheduler. Add host-wide exclusion
   for legacy local-live/CLI/subprocess entry points on the same inference host.
   If integrated inference owns the host, those paths return an explicit busy
   result rather than load competing model copies. They remain usable after
   deliberate shutdown/unload and release. Do not silently stop a field session.
5. **Durable local state.** Use SQLite via the standard library for mission,
   event, action, and evidence metadata; keep media in a configured durable data
   directory outside temp storage. One runtime writes it. Original media is
   immutable; generated views/annotations are separate artifacts.
6. **Incremental interface.** Extend the existing hub WebSocket with negotiated
   JSON mission commands/events. Preserve binary frame layout and legacy manual
   messages. No new HTTP server, message broker, or cloud dependency is required
   for this milestone.
7. **Private sandbox first.** Serve new controls only on loopback or the explicitly
   approved private network with authenticated, server-assigned scopes. Default
   business writes require an operator approval for the exact proposal. Later
   preauthorization must be explicit policy, not an LLM interpretation.

```text
Scout / Ground Control browser
       | existing stream + negotiated mission commands
       v
Bridge hub process: source ownership + transport
       |
       +-- visionbrain mission runtime -- durable state / original evidence
       |           |
       |           +-- existing perception / reasoning tools
       |           +-- approved sandbox business adapter + read-back
       |
       +-- correlated snapshots, events, results, and evidence to both clients

Existing local-live / CLI / batch tools remain separate entry points;
host admission prevents simultaneous competing inference in integrated mode.
```

## 4. Shared contract v1

The names below are proposed additions, not current commands. Freeze their
machine-readable definitions and golden fixtures in CM-02 before client work.
Python definitions belong in VisionBrain; wire serialization belongs in the
bridge and `android/bridge-core`. Do not expose bridge internals to the core.

### Identity, evidence, and time

| Record | Required meaning |
|---|---|
| Mission | `mission_id`, `schema_version`, `revision`, trusted goal, profile ID/version, source binding, state, required items, unresolved reasons, active policy/budgets. |
| Source binding | Server-approved `source_id`, new `source_epoch` for each producer connection, explicit client/inference/archive session associations; a mission may span successive sessions. Client role text is not identity or permission. |
| Observation | `observation_id`, mission/source/epoch/session IDs, analyzed `frame_id`, actual `observed_frame_id` when output is held, dimensions, evidence IDs, detector/tool revision, prompt/configuration revision, observation status. |
| Evidence | Opaque `evidence_id`, original-media hash, source observation, artifact kind, provenance, review state, and durable availability. Crops/annotations reference their original and crop coordinates. |
| Context note | Attributed author/source/time, content, and mission association. It may guide investigation but cannot grant tool authority or verify physical state by itself. |
| Entity match | Candidate observation, proposed business item ID, matching evidence, review/verification basis, and `candidate / verified / unresolved` state. Local track IDs are not business IDs. |
| Action | `action_id`, mission ID, proposal hash, tool/schema version, exact arguments, evidence IDs, policy/approval reference, idempotency key, expected business-record version, receipt, verification result. |

Current `hello.session_id`, Brain session ID, and archive session ID have different
origins; preserve their meanings and record their mapping. Add distinct
`client_session_id`, `inference_session_id`, and `archive_session_id` in the new
contract instead of silently redefining legacy `session_id`. Binary decoding
currently supplies a generic source and no dimensions: obtain approved source
identity from the connection and dimensions from the actual JPEG. Client-provided
binding fields select an existing permitted binding; they do not create authority.

Two concrete remaining bridge checks belong in CM-01/02/05: `frame_buffer.py`
currently computes latency by subtracting raw timestamps from host UTC, invalid
for wrapped binary time; and `archive.py` ordinary result rows allow a payload
`session_id` to override the archive ID, unlike frame/rich-result rows. Define
and test explicit linkage without rewriting old archive identities. Stamp true
receive time at ingestion; archive-worker write time is a separate timestamp.

Capture time may be unknown. Keep raw wire timestamp and its encoding separate
from an optional full UTC capture time; retain clock quality/uncertainty. The
binary u32 timestamp is not full epoch time. Server receive/inference/publish
times are separate, with monotonic timestamps for same-host durations. Never
derive authoritative freshness by subtracting a wrapped client value from UTC.

Define observation statuses at least `observed`, `held`, `unavailable`, and
`failed`. An observed empty detection set differs from no observation. Freshness
limits are profile settings evaluated by the runtime, not invented by a model.
Restart/reconnect makes live freshness unknown until a new accepted observation.

Deduplicate transport using request/event identity and source epoch/frame identity.
A media hash may deduplicate blob storage, but identical pixels at different times
are not automatically the same operational event. Repeated observations of one
item must not increase its inventory count.

### Commands, replies, and replay

Negotiate capability `mission.v1` before using new message types. A client without
it retains manual tools but cannot silently start an autonomous mission.

Proposed envelope:

```json
{
  "type": "mission_command",
  "schema_version": 1,
  "request_id": "client-generated-unique-id",
  "mission_id": "existing-id-or-null-for-create",
  "expected_revision": 4,
  "command": "pause",
  "args": {}
}
```

Commands: `create`, `resume`, `pause`, `cancel`, `get`, `list`, `events_since`,
`add_note`, `attach_evidence`, `approve_action`, and `get_evidence`.
Creation supplies a profile, goal, source binding, and sandbox record reference;
the sandbox reference is required for the checklist profile and omitted for the
perception-only `visual_inspection` profile. That profile also supplies expertise,
mode, and a qualified reasoning model as defined in the controller detail.
The server assigns the mission ID. Creation omits `expected_revision`.
Mutating existing missions requires it; read-only commands omit it. `list`
uses installation scope, not a fabricated mission ID. Define bounded pagination
for mission/event reads and bounded, checksummed media transfer in CM-02.

Every command returns `mission_reply` with the same request ID, mission ID if
assigned, resulting revision, status, and structured result/error. Success means
that command completed; it does not mean the entire mission is complete.
Mutations append durable `mission_event` records with a monotonic per-mission
sequence. `get` returns a snapshot plus its last sequence; `events_since` provides
reconnect catch-up. Define explicit `resync_required` when retained history is
insufficient. Live frames remain latest-only and are not replayed as fresh frames.

Errors include `unauthorized`, `unsupported`, `revision_conflict`, `source_busy`,
`inference_busy`, `evidence_unavailable`, `budget_exceeded`, and `invalid_request`.
Duplicate request ID + identical payload returns the stored reply; the same ID
with changed payload is rejected. Scope request IDs to authenticated principal
and installation. A lost reply must not repeat a business side effect.

### State and execution

Mission states: `created`, `running`, `waiting_evidence`, `waiting_approval`,
`paused`, `completed`, `failed`, `cancelled`. Every transition persists its reason
and increments revision. On recovery, unfinished work becomes paused with a
recovery reason; reconcile pending business receipts before offering resume.
Completed means the goal's verification rule passed, not merely that a tool ran.
`create` persists `created` without starting inference; explicit `resume` starts
or resumes permitted execution. An Inspect photo must have an acknowledged
evidence attachment before resume; a stream detection acknowledgment is not an
evidence receipt. Close-up attachment references the outstanding request and
cannot automatically resume a paused/cancelled mission. A reconnect alone never
resumes execution.

Action states: `proposed`, `authorized`, `requested`, `acknowledged`, then
`verified`, `failed`, or `unresolved`. An ambiguous timeout becomes unresolved;
query the destination by idempotency key before any retry. Approval is bound to
the exact arguments/proposal hash, evidence set, and expected business version;
changed arguments or stale preconditions require a new proposal/approval.

At every tool dispatch and result commit check scope, current mission state,
source freshness, budget, and a server-issued execution-generation token. Client
mutations use expected revision; a cycle's own valid commits may advance revision
without cancelling that cycle. Pause/cancel, source or relevant context changes,
and manual override invalidate its generation, as detailed in the controller spec.
Suggested starting bounds for the fixture: one active
investigation, ten tool calls, 180 seconds of active investigation time, and an
explicit timeout for human evidence/approval waits. CM-00 records final values.
A timeout stops new dispatch; it must not claim native inference already running
has been safely interrupted. Late results from an obsolete execution generation are retained
as historical evidence but cannot resume or mutate a cancelled mission.

The controller may retarget a bounded prompt set, ground an expression, crop an
identified observation, run supported OCR, inspect sandbox records, or request a
human close-up. It cannot execute arbitrary shell commands, arbitrary URLs, or
unrestricted writes. Vision/OCR/notes are untrusted input. Existing manual
Ask/Report grounding rules remain unchanged: a separately labeled investigation
mode can acquire new scene evidence; it must not relax ordinary report rules.

### Source controls and scheduling

Route mission and legacy manual controls through one bridge-owned arbiter.
Manual pause/override wins, suspends the investigation, and invalidates its
temporary control lease. Temporary prompts carry mission/revision, purpose,
expiry, and a return configuration; expiry must not overwrite a newer manual
choice. Unsupported tools fail before dispatch, not after a misleading UI change.

Configure no safety-critical watches in v1. If required watches are configured,
reject investigations whose tool/model allocation cannot preserve their declared
cadence. Expensive archive work and evidence encoding stay off the frame hot path;
bound their queues separately. If evidence cannot be persisted, show unavailable
and prevent dependent verification/action; never substitute a different frame.

### Sandbox business adapter

Provide `read_checklist(record_id)` and
`record_verification(record_id, item_ids, evidence_ids, expected_version,
idempotency_key)`. Return a durable receipt and resulting record version; verify
with a read-back. The fixture uses a local transactional sandbox, not real stock.
Return conflicts explicitly. No arbitrary JSON overwrite or count-to-stock rule.

Verify item identity with supported readable identifiers plus record matching,
or an explicitly attributed operator confirmation. Detector confidence alone
does not establish exact identity. Missing/occluded items remain unresolved;
the workflow may finish a reviewed partial checklist only if that is the stated
goal, never silently convert a full-readiness goal into partial success.

### Access, retention, and recovery

Assign scopes server-side: mission/evidence read, mission control, and action
approval. Viewer credentials cannot mutate; source credentials cannot approve.
Every transport, media transfer, and reconnect enforces those scopes. Do not log
tokens or send host file paths as evidence identifiers. Resolve media IDs under
the configured evidence root and reject traversal or cross-installation access.

Set storage limits before enabling capture. Quota exhaustion produces an explicit
state and blocks evidence-dependent writes; it must not silently erase referenced
evidence. Supply a manual backup/restore procedure and a retention/export policy.
No automatic destructive cleanup is part of the first fixture.

## 5. Assignable punch list

All paths below are ownership guidance. New module names are proposed; keep the
public interface small and test through it. Existing source/tests take priority
over diagrams in old notes.

| ID / owner | Work and source surfaces | Dependency | Completion evidence |
|---|---|---|---|
| CM-00 / integration lead | Ratify this scope/topology, record paired baseline commits plus relevant dirty diffs, runtime budgets, fixture, and hardware acceptance thresholds. Resolve existing spec conflicts listed below. | None | One signed decision record, owned tickets, and a reproducible baseline; no disputed semantics left for teams to guess. |
| CM-01 / core + bridge QA | Review and verify existing timestamp, FastScan, native-tool, and pilot-eval changes. Reproduce remaining binary-age and archive-ID linkage issues. Classify failures against actual source; do not dismiss them as stale without evidence. | CM-00 | Focused and affected regression results, plus configured-backend conversation check when authorized. Existing work integrated without duplicate patches. |
| CM-02 / core contract lead + bridge reviewer | Define versioned records, envelopes, transitions, capability negotiation, limits, evidence transfer, and golden Python/Kotlin fixtures. Suggested core `mission_contracts.py`; bridge `protocol.py` and protocol docs. | CM-00 | Same golden cases accepted/rejected by both languages, including wrap/unknown time, duplicate IDs, and revision conflicts. Frozen contract available to both teams. |
| CM-03 / core persistence | Implement `mission_store.py` using durable SQLite metadata/events and immutable original-media storage. Reuse archive/session/evidence concepts. | CM-02 | Restart, duplicate ingest, missing media, quota, traversal, and backup/restore tests pass. State mutation and event append are atomic. |
| CM-04 / core runtime + bridge execution | Implement one host inference admission gate and serialized integrated executor around existing model calls. Cover bridge, local-live, and legacy job/CLI entry points; show busy instead of starting competitors. | CM-00, CM-01 | Tests prove exclusion across processes and safe release after process exit. Field capture remains responsive; pause/timeout does not falsely claim native cancellation. |
| CM-05 / bridge transport | Add negotiated mission command/reply/event wiring, authenticated scopes, single-producer ownership/epoch, control arbitration, and evidence receipt/delivery. Suggested thin `missions.py`; reuse `server.py`, `archive.py`, `cadence.py`, `bridge-core`. | CM-02, CM-03, CM-04 | Scout/browser access the same mission; competing sources rejected; reconnect catches up; stale and mismatched evidence cannot verify a result. |
| CM-06 / core autonomy | Implement `mission_runtime.py` with a deterministic state machine and a bounded planner using existing visual tools. Read-only investigations first; inject replay and field adapters. | CM-01, CM-02, CM-03, CM-04 | A fixture uncertainty triggers a useful observation request; budgets, cancel, manual override, malformed tools, and malicious OCR/notes pass the tests below. |
| CM-07 / business integration | Implement the transactional checklist adapter, exact-proposal review/approval, idempotency, version checks, and outcome read-back. | CM-03, CM-06 | Lost acknowledgment, duplicate delivery, stale approval, conflict, and failed read-back cannot produce duplicate writes or false completion. |
| CM-08 / Ground Control + Scout UI | Add create/select mission, status, candidate/verified/unresolved items, evidence view, requested close-up, review/approve, pause/cancel, and reconnect state. Use existing UI grammar; no redesign. | CM-02 fixtures; integrate after CM-05–07 | Both clients show the same revision and action outcome. No connected-looking stale state or unsupported controls; view-only access is enforced. |
| CM-09 / independent evaluation | Extend/reuse `pilot_eval.py` with mission outcome fixtures and real timing instrumentation; add cross-repo replay tests. | CM-05–08 | Every software gate below passes, with artifacts. M2 trial measures accuracy, latency, memory, and degradation against CM-00 thresholds. |
| CM-10 / release/installation | Provide paired version manifest, explicit data/host configuration, one documented integrated startup/shutdown, capability/health report, rollback and restore instructions, and per-mode standalone smoke checks. | CM-09 | Fresh private installation or restored test installation completes the acceptance scenario; no auto-deploy, hidden model download, or port-killing launcher. |

- [ ] CM-00 baseline and decisions ratified
- [ ] CM-01 existing correctness work verified
- [ ] CM-02 shared interface and golden fixtures frozen
- [ ] CM-03 durable state/evidence ready
- [ ] CM-04 inference admission/execution ready
- [ ] CM-05 bridge connection/control/evidence ready
- [ ] CM-06 adaptive investigation ready
- [ ] CM-07 sandbox business action ready
- [ ] CM-08 both operator interfaces ready
- [ ] CM-09 independent replay and hardware gates passed
- [ ] CM-10 reproducible installation and rollback demonstrated

### Existing spec conflicts to settle in CM-00

- The UI rework plan describes both parked controls on reconnect and no retry
  queue. New mission commands use persisted request identity and reconciliation;
  ephemeral live frames remain latest-only. Legacy stale control replay is not
  authorized. Write the resolved behavior into the affected interface docs.
- The UI plan's counts/white detector agreement must not be represented as
  verified inventory or ground truth. Preserve its visual language while making
  business verification a separate explicit state.
- The simplification work scopes ordinary local Ask/Report to current armed
  evidence. Preserve it; adaptive scene investigation is a distinct mode.
- Fast-pipeline ideas for parallel Falcon and cached propagation are not proof
  of thread safety or current-frame accuracy. Preserve serialized execution in
  this milestone unless an independently measured change passes the same gates.
- Root/older notes and repo docs disagree about hardware and old paths. The
  bridge's current [inference-host instructions](../visionBrain-bridge/docs/INFERENCE_HOST.md)
  govern field trials: designated M2, pinned environment, explicit operator
  authorization. No VPS move, model upgrade, or automatic restart is implied.

## 6. Acceptance gates

Each case needs a recorded input, expected result, actual result, and evidence
artifact. Deterministic/replay gates require zero invariant violations across the
suite; that is not a claim of zero real-world model errors.

| Gate | Test | Required result |
|---|---|---|
| A01 | Create in Scout, read/review in Ground Control | Same mission ID, authoritative revision, items, and evidence. |
| A02 | Unreadable label followed by requested close-up | First state unresolved; new evidence linked to the request; verification only after sufficient identity evidence. |
| A03 | Item leaves view; misleading shift note says shipped | No invented transfer or inventory deduction. Note remains attributed context. |
| A04 | Old/held detections, unknown capture time, mismatched snapshot | Explicit stale/unknown/unavailable state; no current verified claim from mismatched media. |
| A05 | Producer B connects while A owns source; A reconnects/reset counter | B rejected; A has a new epoch; no cross-source evidence or reused frame identity. |
| A06 | Same request delivered twice; same ID with different payload | One mutation for identical replay; changed payload rejected. |
| A07 | Two clients mutate revision N; approval refers to superseded proposal | One accepted mutation; conflict/new approval required for stale work. |
| A08 | Destination commits but reply is lost | Read receipt/destination state by idempotency key; no blind duplicate write. |
| A09 | Action acknowledged but read-back missing/wrong | Action unresolved/failed; mission not completed. |
| A10 | Process restart between mutation/event publication | Durable state and event agree; snapshot/cursor recovery; unfinished mission paused; no automatic replay of an uncertain write. |
| A11 | Manual pause/cancel while reasoning runs; result arrives late | No new action; late evidence is historical; old prompt lease cannot overwrite manual control. |
| A12 | OCR/note attempts to issue commands or alter policy | Treated as data; no scope change, arbitrary tool, or unapproved write. |
| A13 | Quota/disk failure, missing media, unauthorized/path-traversal request | Explicit failure; no fabricated evidence, unsafe path access, or dependent action. |
| A14 | Slow inference/viewer while source streams | Bounded frame/encoding queues, truthful age and drops, responsive control connection. |
| A15 | Legacy GPU job/local-live start while integrated runtime owns host | Explicit busy; no competing model load. Standalone operation succeeds after legitimate release. |
| A16 | Malformed/unsupported tool call, exhausted budget, unsupported capability | Structured failure or bounded recovery; no unbounded loop or silently ignored side effect. |
| A17 | Restore a backup into a separate test data directory | Mission/evidence/action linkage intact; unresolved writes remain unresolved until checked. |
| A18 | Model-backed fixture and representative authorized footage | Report false verifications, missed items, unresolved rate, review effort, task completion, and measured latency/memory against predeclared thresholds. |

Measure capture-to-receive only when clocks permit it; separately measure queue,
decode, inference, publish, UI display, and business-action durations. Distinguish
warm/cold runs and model invocation count. Report p50/p95 with sample count and
hardware/configuration. A sampled media-timeline offset is not end-to-end latency.
Choose any hardware purchase only after this bottleneck breakdown exists.

### Verification commands and evidence discipline

Suggested existing model-free regression entry points (not run to write this spec):

```sh
# From VisionBrain, using its prepared environment:
.venv/bin/python -m pytest tests/test_agent_loop.py tests/test_frame_selector.py tests/test_pilot_eval.py -q
.venv/bin/python -m pytest tests/ -q

# From visionBrain-bridge, using its prepared environment:
bridge/.venv/bin/python -m pytest tests/test_protocol.py -q
bridge/.venv/bin/python -m pytest tests/ -q

# From visionBrain-bridge/android, with an existing configured Android SDK:
./gradlew :bridge-core:testDebugUnitTest :scout:testDebugUnitTest :scout:assembleDebug
```

Add new module/integration tests beside these and keep them runnable with injected
models, clocks, and sandbox tools. Browser/Android checks must cover real
interaction and reconnect state, not just DOM presence. Hardware tests are a
separate explicitly authorized gate on the designated M2; do not install weights
on the coding machine or infer deployment readiness from mocked tests.

## 7. Team/agent dispatch and release rule

Start CM-00/01 first. Once CM-02 is frozen, persistence and execution work can run
in parallel, and UI work can use the golden fixtures. Assign only one owner per
shared file; use named core/bridge tasks for cross-repo edits. Independent QA
reviews the assembled workflow, not just each team's mocks. Company marketing
and sales receive only the verified capability/release report, not unchecked
checkboxes as customer promises.

Give each implementation agent: ticket ID, allowed files, frozen contract version,
dependencies, explicit prohibited actions, acceptance gate IDs, and required
evidence. Finish with changed files, test results, unresolved risks, and handoff
conditions. Runtime agent role instructions remain in the
[shared fleet baseline](BUSINESS_CAPABILITY_RECON.md#ai-agent-fleets-company-versus-customer-operation);
tool scopes and policy enforcement must exist in code.

Release only when CM-01–10 are supported by their recorded evidence and A01–18
pass the applicable gates. Any unavailable hardware or undecided customer
threshold stays an open gate. The next increment is selected from measured needs:
another business adapter/profile, multi-camera routing, model evaluation/training,
or remote compute—not assumed to be all of them at once.
