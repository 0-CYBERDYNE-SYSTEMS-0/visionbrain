# CM-00 Decision Record — platform-first productization

> **2026-10-07 scoped amendment:** [Expertise-only Watch / Jev](EA_JEV_DECISION_RECORD.md) authorizes the optional hosted decision service and canonical mission changes approved by the user. Existing provenance, runtime ownership and no-deployment constraints remain binding.

Date: 2026-09-15  
Authorized by: founder directive in the working session, acting as product owner and integration lead  
Status: ratified for the next development cycle; implementation evidence remains required

This record closes the decision portion of CM-00. It does not claim that the
software is integrated, release-ready, customer-ready, or deployed. The paired
baseline is recorded in [PAIRED_BASELINE_MANIFEST.md](PAIRED_BASELINE_MANIFEST.md),
and the working-tree disposition is recorded in
[CM00_DISPOSITION.md](CM00_DISPOSITION.md).

## Founder intent

Build and prove the reusable visual-intelligence system before selecting a real
customer niche. Synthetic and permissioned recorded mission packs are the first
development and demonstration inputs. A customer or paid pilot is not a
prerequisite for completing the internal platform slice.

The product is not a report generator. Its value is the inspectable pipeline:

```text
camera or deterministic replay
  -> source-owned capture and transport
  -> immutable original evidence
  -> detection / boxes
  -> segmentation / masks
  -> tracking / temporal continuity
  -> evidence-linked candidate findings and uncertainty
  -> attributed human review
  -> durable visual evidence packet
```

Reports are one output of this pipeline. The boxes, masks, tracks, provenance,
uncertainty, review history, and repeatable execution must remain visible in the
product and test evidence.

## Ratified architecture and scope

1. **One integrated runtime authority.** In integrated mode, the existing
   bridge hub owns source connection, transport, source epochs, and control
   arbitration. An importable VisionBrain mission runtime inside that process
   owns mission state and decisions. Scout and the Ground Control browser are
   clients of the same authority. Ground Control's HTTP process must not create
   a second mission controller or mission database.
2. **Scout is the first live capture client.** One supported Android phone is
   the first live camera surface. Drone, CCTV, Pi, robot, and imported-video
   adapters must later enter through the same source contract; they do not get
   separate mission architectures.
3. **Replay and live capture use the same downstream interface.** A deterministic
   replay adapter is built first for repeatable work. Replacing replay with Scout
   capture must not change mission, observation, evidence, finding, review, or
   packet semantics.
4. **VisionBrain owns visual intelligence and evidence semantics.** Detection,
   segmentation, tracking, temporal association, model orchestration,
   uncertainty, finding construction, provenance, review state, and packet
   generation are canonical VisionBrain responsibilities. Bridge transports
   and schedules them; clients present and control them.
5. **Product mode and Lab mode are two views of one run.** Product mode supports
   mission creation, capture, findings, human review, and export. Lab mode exposes
   frames, boxes, masks, track IDs/history, model/backend identity, confidence
   components, timing, resource use, errors, and evidence lineage. Lab mode is
   observability, not a second processing pipeline.
6. **Freeze the released contract subset first.** CM-02 freezes Observation,
   Evidence, Finding/Review, Inspection Packet, Runtime Capability, and
   Perception Adapter records in an MLX-free module with golden Python, Kotlin,
   and browser fixtures. The broader `mission.v1`, adaptive Inspect, business
   actions, and Watch behavior return after this vertical slice works.
7. **Durable local-first state.** SQLite holds mission/event/review metadata;
   immutable originals and separate derived artifacts live in a configured
   durable evidence root. No new backend service, broker, cloud dependency, or
   repository merger is authorized for this cycle.
8. **One inference owner per host.** Integrated inference is serialized behind a
   host-wide admission mechanism. Separate live, CLI, batch, and bridge processes
   may not load competing model copies; they return a truthful busy state.

## Synthetic demonstration portfolio

These are mission profiles over one platform, not separate applications or
separate forks.

### Pack 1 — facility condition inspection (first integration fixture)

- Simulated organization, site, rooms/zones, visit, and checklist.
- Visible examples: wall damage, water stain, blocked path, missing fixture,
  surface condition, and equipment in the wrong zone.
- Includes successful findings, successful-empty views, insufficient-detail
  views, an unreadable label requiring a closer view, stale evidence, duplicate
  delivery, and an explicit processing failure.
- Includes still images plus a short sequence that exercises stable track IDs,
  disappearance/reappearance, and before/after comparison.

### Pack 2 — commercial service verification

- Before/after capture of zones and surfaces.
- Exercises masks, change evidence, unresolved conditions, reviewer correction,
  and a service-evidence packet.
- Must not equate an empty detector response with completed work.

### Pack 3 — warehouse safety walkthrough

- Simulated blocked aisle, misplaced object, missing visible safety item, and
  repeated observation along a walkthrough.
- Exercises boxes, masks, temporal tracks, zone context, attributed notes, and
  a reviewed action list without making regulatory or public-safety claims.

Pack 1 is the integration gate. Packs 2 and 3 are added as configuration and
fixture data only after Pack 1 completes the same end-to-end path. No
vertical-specific product branch is authorized.

## Canonical states and binding truth rules

- Observation: `observed | held | stale | failed | unavailable | capture-time-unknown`.
- Visual finding: `candidate | supported | unresolved`.
- Identity: `unknown | candidate-match | externally-verified`.
- Review: `pending | accepted | corrected | rejected`.
- Packet: `draft | approved | exported | superseded`.
- Business action: `proposed | approved | attempted | confirmed | unresolved | failed`.

A model score, detector agreement, stable track, or empty result cannot silently
promote another layer. Every transition records actor, time, prior revision,
reason, and evidence references. Original media remains untouched; annotations,
masks, crops, and reports are derived artifacts linked by transforms and hashes.

## Initial engineering bounds

These bounds control development behavior; they are not customer-performance or
accuracy claims.

- One active/waiting mission owns the live source at a time.
- One active investigation at a time.
- At most 10 tool calls and 180 active seconds per investigation.
- Human-evidence wait expires after 60 minutes and pauses rather than inventing
  completion.
- Evidence-root development quota: 100 GB; exhaustion becomes an explicit
  failure and never triggers silent deletion of originals.
- Unsupported positive business conclusions: zero tolerated in deterministic
  fixtures.
- Software invariant violations listed below: zero tolerated.
- Record cold/warm p50 and p95 by stage with sample counts on the designated M2;
  do not impose or advertise a production latency target until the first
  reproducible benchmark establishes the baseline.
- Record actual checkpoint revision, backend, configuration, toolchain, and host
  for every model-backed acceptance run. A marketing model name is insufficient.

Model-quality thresholds for real footage are deliberately not invented here.
CM-09 must establish a labeled benchmark, report misses, false candidates,
unsupported positives, unresolved rate, reviewer corrections/time, and resource
use, then propose thresholds for a separately recorded decision.

## Resolved specification conflicts

- Persisted mission commands use request identity and reconciliation after
  reconnect. Ephemeral frames remain latest-only. Legacy stale controls are not
  replayed.
- Counts, detector agreement, and tracks are visual evidence, not verified
  inventory, identity, compliance, or ground truth.
- Ordinary Ask/Report remains bounded to currently armed evidence. Adaptive
  investigation is a distinct later mode.
- Inference remains serialized until measurements and regression evidence
  justify a concurrency change.
- The designated M2 and its pinned environment govern model-backed trials. No
  VPS move, model upgrade, automatic deployment, or automatic restart is implied.
- The earlier requirement for a named prospect and a single commercial niche is
  replaced for this development cycle by the synthetic mission portfolio above.
  A real D1 prospect demonstration still requires permissioned representative
  inputs and a prospect-specific acceptance record.

## Immediate implementation order

Agents must not jump directly into a broad UI redesign or niche customization.

### Step 1 — CM-01: verify the baseline

- Re-run focused and full VisionBrain, bridge Python, and Android tests.
- Reproduce and classify timestamp wrap/age, archive/evidence identity,
  duplicate delivery, reconnect, and model-admission issues against current code.
- Record changed files, commands, results, and unresolved failures. Do not add a
  duplicate fix where current code already addresses the issue.

### Step 2 — CM-02: freeze the smallest shared contract

- VisionBrain contract owner defines the MLX-free canonical records and state
  transitions.
- Produce versioned valid and invalid golden fixtures, including epoch reset,
  unknown capture time, duplicate IDs, revision conflict, geometry bounds,
  capability mismatch, successful-empty, failure, and missing evidence.
- Bridge Python, Kotlin `bridge-core`, Scout, and browser tests must consume the
  same fixture revision. Contract changes land before dependent implementation.

### Step 3 — build deterministic Pack 1 replay

- Create a checksummed manifest containing synthetic/authorized media, expected
  observations, expected temporal relationships, expected review decisions, and
  expected packet contents.
- Replay enters through the same source-adapter boundary planned for Scout.
- Expected outputs test contract and workflow truth; synthetic success is not an
  accuracy claim about uncontrolled real scenes.

### Step 4 — CM-03 and CM-04 in parallel after their dependencies

- CM-03 implements atomic durable mission/event/review metadata, immutable
  originals, separate derived artifacts, quotas, export manifests, backup, and
  restore.
- CM-04 implements host-wide inference admission and a serialized executor
  covering integrated, local-live, CLI, and batch entry points.

### Step 5 — CM-05: connect the authority

- Add negotiated mission commands/events, authenticated scopes, one-producer
  source epochs, control arbitration, bounded evidence delivery, receipts, and
  reconnect catch-up to the existing bridge hub.
- Preserve the existing binary frame layout; additive protocol work must keep
  protocol documentation and shared fixtures synchronized.

### Step 6 — CM-08: complete Product and Lab UI paths

- Ground Control: mission list/detail, evidence viewer, annotated playback,
  box/mask/track controls, provenance inspector, pending-review queue, attributed
  accept/correct/reject, packet preview/export, and Lab telemetry.
- Scout: create/select mission, guided capture, upload/processing state,
  preliminary overlay where supported, recapture, reconnect truth, and reviewed
  results.
- Both clients display the same mission ID and authoritative revision. Neither
  client fabricates connected/current/approved state.

### Step 7 — prove live Scout equivalence

- Run one supported Scout phone through the exact Pack 1 downstream path.
- Demonstrate immutable original capture, boxes, at least one mask-capable path,
  temporal track evidence where the input supports it, uncertainty, attributed
  review, packet export, disconnect/reconnect, and unavailable-evidence behavior.
- Manual steps must be disclosed until automated durable transfer is proven.

### Step 8 — CM-09/10: evaluate and package

- Execute deterministic gates and model-backed runs separately.
- Record actual configuration, hashes, uncut/logged runs, stage timing, memory,
  drops, failures, review work, and packet results.
- Prove installation, startup/shutdown, backup/restore, and rollback on the
  supported private configuration. Then add Packs 2 and 3 without changing the
  canonical pipeline.

## Acceptance gates for the platform slice

The existing A01-A18 gates in `NEXT_DEVELOPMENT_SPEC.md` remain authoritative.
For this cycle, completion additionally requires one recorded run showing:

1. Pack 1 replay and Scout capture create the same canonical object types.
2. Exact originals, hashes, geometry transforms, boxes, available masks, and
   track lineage are inspectable from the resulting findings.
3. Candidate, supported, unresolved, and review states remain distinct.
4. A reviewer can correct or reject a candidate without rewriting raw model output.
5. Packet export is reproducible and traceable to immutable evidence.
6. Reconnect, stale input, missing evidence, duplicate input, disk/quota failure,
   and competing inference/source ownership are represented truthfully.
7. Lab mode exposes the actual processing run; it does not substitute simulated
   telemetry for missing capability.
8. The run can be repeated from documented commands after restart/restore.

Zero-tolerance software invariants are: wrong-source/epoch evidence, mismatched
annotation/original, duplicate durable mutation, unauthorized review/control,
stale data shown as current, false business completion, and competing model-load
ownership.

## Explicitly deferred

- Selecting or hard-coding a permanent customer niche.
- Production business-system writes and autonomous physical actions.
- Watch mode, unrestricted agents, multi-camera/multi-tenant operation, and
  cross-camera identity.
- Automatic aircraft or robot motion and public-safety/compliance claims.
- Repository merger, third backend service, cloud/VPS migration, and CI-driven
  deployment or M2 restart.
- Custom training or new model/platform integrations without a measured failure
  in the canonical evaluation.
- Splitting the synthetic business profiles into separate applications.

## Agent execution contract

Every assigned task must state: CM/PB ticket ID, repository and allowed files,
frozen contract/fixture version, dependencies, prohibited actions, applicable
acceptance gates, and required evidence. Every handoff must report changed files,
test commands/results, remaining risks, and the exact condition that unblocks the
next owner.

No push, merge, release tag, deployment, M2 restart, model download/upgrade, or
customer-facing readiness claim is authorized by this record. Cross-repository
protocol work requires linked companion changes and the same golden fixture
revision. Existing unrelated working-tree changes remain owned by their authors.

## Supersession note

For this development cycle, this record supersedes only the conflicting
commercial-scope statements in `PRODUCTIZATION_EXECUTION_PLAN.md`, its D1-first
dispatch language, the interactive `cm00-ratification.html` prospect requirement,
and the bridge PB-00 statement that cleaning/facilities is the single actively
productized hypothesis. All safety, evidence, provenance, state-separation,
Git/release, and no-deployment constraints remain binding. The bridge team must
record the corresponding PB-00 amendment before treating its old commercial
hypothesis as current.
