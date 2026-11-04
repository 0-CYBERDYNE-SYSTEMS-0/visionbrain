# VisionBrain productization execution plan

Date: 2026-09-13  
Status: proposed execution overlay; requires CM-00 ratification  
Primary outcome: make VisionBrain ready to support honest demonstrations, controlled internal experiments, and one Scout-first paid pilot workflow

## Authority and relationship to existing specifications

This document owns productization priority, team sequencing, demo readiness, and release evidence for the next delivery cycle. It does not replace:

- [NEXT_DEVELOPMENT_SPEC.md](NEXT_DEVELOPMENT_SPEC.md), which remains authoritative for Connected Mission runtime semantics, CM ticket IDs, shared contracts, and A01–A18 acceptance gates;
- [BUSINESS_CAPABILITY_RECON.md](BUSINESS_CAPABILITY_RECON.md), which remains the capability and business-evidence reference;
- [docs/architecture-review-20260912-visualbrain.html](docs/architecture-review-20260912-visualbrain.html), which remains the architectural review;
- the bridge repository's `ADAPTIVE_MISSION_CONTROLLER_SPEC.md`, which remains authoritative for proposed Inspect/Watch behavior and qualification.

If these documents conflict, stop and resolve the conflict in CM-00. Do not silently invent new semantics.

## Product decision

VisionBrain is the reusable visual-intelligence platform and applied lab foundation. The first commercial capture surface is the Scout Android phone app.

The next product slice is:

```text
one supported Scout phone
    → one bridge runtime on the designated M2
    → VisionBrain perception and reasoning components
    → exact source evidence plus candidate findings
    → named human review
    → durable, exportable inspection packet
```

This slice is deliberately narrower than Connected Mission v1. Completing it does not complete CM-06 adaptive investigation, CM-07 business actions, Watch mode, multi-camera operation, or the full A01–A18 release.

The company may explore cleaning/facilities, recorded-site review, aerial inspection, warehouse checklists, robotics perception, CCTV, and other smart-camera problems. Only one funded workflow enters delivery at a time.

## Product and technical positioning

VisionBrain should be presented as:

> An applied visual intelligence platform that turns supported camera inputs into evidence-backed findings through detection, segmentation, tracking, visual reasoning, and human review.

The report is an output. The core value engine is:

```text
source pixels
  → boxes / labels
  → masks / spatial extent
  → tracks / temporal continuity
  → cross-engine evidence and uncertainty
  → task-context reasoning
  → inspectable artifacts
  → human decision or bounded next observation
```

Do not collapse those layers into a single “AI confidence” number. Preserve the evidence vector and its provenance.

## Current readiness

### Implemented foundation

- Image detection, segmentation, OCR, SAM image/video work, tracking, video analysis, FastScan, and bounded single-image agent tools.
- FastAPI Ground Control application with static browser UI, job APIs, SSE, and a standalone live engine.
- Reusable geometry, detection normalization, tracking, zones, model residency, and VLM registry components.
- Bridge imports of canonical VisionBrain detection, tracking, VLM, and model-host code.
- Scout capture, overlays, snapshots, and journaling exist in the companion repository.

### Not yet established as a customer-ready product

- Durable shared inspection/mission state across Scout and Ground Control.
- Immutable untouched originals linked to every released finding.
- Complete candidate/reviewed/unresolved and approval semantics.
- Reliable automatic Scout-to-Ground-Control evidence delivery.
- Host-wide inference exclusion across separate processes.
- One frozen wire/mission contract proven across Python, Kotlin, and browser consumers.
- Repeatable installation, backup, restore, rollback, and paired release evidence.
- Accuracy, latency, stability, or economics on a selected customer's task.

### Known truth that must appear in every release record

- Ground Control job state is process memory and current media work directories are temporary.
- VisionBrain's local live engine and the bridge hub remain separate orchestration paths.
- `model_host.HOST` shares residency within one process, not across processes.
- The live VLM registry key `gemma` currently resolves to its actual configured checkpoint; release artifacts must record the checkpoint and backend rather than relying on an ambiguous marketing model name.
- Successful empty, unsupported, failed, timed-out, stale, and unknown are different outcomes.

## State and confidence model

The released workflow must keep these concepts separate:

| Layer | Allowed state examples | Meaning |
|---|---|---|
| Observation | observed, held, stale, failed, unavailable, capture-time-unknown | What data actually arrived and how fresh it is |
| Visual finding | candidate, supported, unresolved | What the visual evidence supports |
| Identity | unknown, candidate match, verified against an external identifier | Whether the business entity is known |
| Review | pending, accepted, corrected, rejected | What an authorized human decided |
| Packet | draft, approved, exported, superseded | Whether an output may be delivered |
| Business action | proposed, approved, attempted, confirmed, unresolved, failed | What happened outside the vision system |

Rules:

1. A high model score does not create `supported` without defined task evidence.
2. Cross-model agreement is supporting evidence, not ground truth.
3. A stable track does not establish global identity.
4. An empty detection list does not create inspection approval.
5. Human acceptance of a report does not retroactively change raw model output.
6. Business identity verification and visual support are distinct.
7. Every transition records actor, time, prior revision, reason, and evidence references.

## Required shared interfaces — CM-02

Freeze the released subset before client implementation. Python definitions belong in an MLX-free VisionBrain module; Kotlin and browser implementations conform through shared golden fixtures.

### Observation record

- server-approved source ID and source epoch;
- frame/observation ID scoped to the epoch;
- dimensions, encoding, and original SHA-256;
- raw wire timestamp plus interpreted capture time or explicit unknown;
- receive time and processing times;
- configuration/model identity;
- observed/held/stale/failed/unavailable status;
- parent request/inspection ID when applicable.

### Evidence record

- immutable original-media ID and hash;
- separate annotation, mask, crop, and report artifacts;
- transform metadata connecting rendered artifacts to the original;
- durable receipt distinct from stream acknowledgment;
- bounded checksummed transfer;
- explicit missing-media, quota, and storage-failure states.

### Inspection packet product layer

- customer/site/visit references;
- inspection template and version;
- required and received views;
- linked candidate/supported/unresolved findings;
- attributed human decisions;
- remaining actions or missing evidence;
- immutable export ID, revision, and manifest.

Do not rename canonical mission/finding states merely to fit UI copy. The packet is a product-layer view over canonical records.

### Runtime capability record

- contract, VisionBrain, bridge, Android, and UI versions;
- actual backend and checkpoint revision;
- available versus task-qualified engines/tools;
- supported geometry and limits;
- source owner and inference owner;
- structured busy, unsupported, degraded, and unavailable reasons.

### Perception adapter contract

- actual image input, not a path represented as text;
- typed bounded results with provenance;
- distinct successful-empty, unsupported, failed, and timed-out outcomes;
- artifacts and transforms needed to render boxes/masks correctly;
- no business conclusion embedded in detector labels.

## Assignable execution plan

### P0 — baseline, contract, evidence, and release safety

| ID | Owner | Work | Dependencies | Acceptance evidence |
|---|---|---|---|---|
| VB-P0-01 / CM-00 | Integration lead | Ratify the Scout evidence-packet slice, selected demo workflow, input/output, reviewer, exclusions, budgets, actual model/configuration, and threshold-setting process. | None | Signed decision record distinguishes assisted demo, customer pilot, Inspect, Watch, and business action releases. |
| VB-P0-02 / CM-01 | Core + bridge QA | Inventory and classify all modified/untracked work. Verify existing timestamp, FastScan, native-tool, pilot-eval, evidence, and import changes before adding overlapping fixes. | CM-00 | Reproduction notes and focused regression results; unresolved failures have owners. |
| VB-P0-03 / CM-02 | Contract owner + bridge reviewer | Define the released observation/evidence/review/runtime subset in an MLX-free module and freeze golden Python/Kotlin/browser valid and invalid fixtures. | CM-00 | Same fixtures accepted/rejected across languages, including timestamp wrap, epoch reset, unknown time, duplicates, bounds, capability mismatch, and revision conflict. |
| VB-P0-04 / CM-03 | Persistence owner | Implement durable metadata/events, immutable originals, separate derived artifacts, review records, export manifests, quotas, and restore. | CM-02 | Restart and restore preserve hashes/linkage; missing or mismatched media blocks supported/released claims. |
| VB-P0-05 / CM-04 | Runtime owner | Implement a host-wide inference admission gate covering bridge, local live, CLI/batch jobs, ASK/REPORT, and model loads. | CM-00, CM-01 | Competing process returns structured busy; field capture remains responsive; ownership releases safely after real completion or process exit. |
| VB-P0-06 / CM-08 | Ground Control owner | Add the minimum evidence review and packet export surface using existing UI grammar; no redesign. | CM-02 fixtures; integrate with CM-03/05 | Reviewer sees exact originals, findings, uncertainty, revision, and release state; stale or unavailable data never looks current. |
| VB-P0-07 / CM-09 | Evaluation owner | Extend evaluation from model events to the full inspection workflow and actual operator timings. | Contract and integrated slice | Recorded dataset/results include misses, false candidates, unsupported positives, unresolved rate, review corrections/time, packet completion, and failures. |
| VB-P0-08 / CM-10 | Release owner | Create reproducible paired release manifest, installation/restore evidence, actual model records, operating guide, and rollback. | Above P0 work | Fresh or restored private installation completes the scoped demo with no undocumented patch/download/restart. |

### P1 — adaptive Inspect after the assisted packet works

| ID | Owner | Work | Acceptance evidence |
|---|---|---|---|
| VB-P1-01 / CM-06 | Runtime/reasoning owner | Deterministic Inspect lifecycle; one validated tool action at a time; bounded crops/OCR/close-up requests; explicit budgets/cancel/manual precedence. | AM-01–04 and applicable Inspect portions of AM-07–16 pass; Watch-specific gates remain open. |
| VB-P1-02 | Model qualification owner | Qualify one actual backend/checkpoint for one task profile before enabling alternatives. | Held-out still/clip suite records schema validity, action relevance, unsupported positives, latency, memory, and model calls. |
| VB-P1-03 / CM-05/08 | Cross-repo UI owners | Connect objective → acknowledged evidence → grounded result → requested better evidence → review in Scout and Ground Control. | Both clients show the same mission/revision/evidence; insufficient evidence remains unresolved. |
| VB-P1-04 | Pipeline API owner | Replace web-to-CLI argv coupling only for the funded paths with typed options/results. | Web and CLI tests share the same callable pipeline; no behavior regression in preserved CLI. |

### P2 — only after repeat paid demand

- Watch mode with paced triggers, leases, freshness, no-op unchanged plans, and manual precedence.
- One demanded business-system adapter with proposal/approval/idempotency/read-back.
- Additional camera adapters through the shared source contract.
- Partner installation tooling and remote diagnostics.
- Broader model adapters only when a measured task failure justifies them.

## Demonstration levels and gates

### Level D0 — internal component demonstration

May show boxes, masks, tracks, OCR, questions, and reports on authorized media. Must identify actual source, model/backend, configuration, date, and limitations. This does not support customer workflow claims.

### Level D1 — founder-assisted prospect demonstration

Required:

1. one named customer problem and defined visual target;
2. permissioned representative input;
3. actual supported Scout phone and designated M2, or an explicitly labeled recorded-input path;
4. preserved original, rendered output, and uncut/logged run;
5. successful finding, successful-empty, insufficient-detail, and failure/degraded examples;
6. every manual step disclosed;
7. no implied autonomous Scout/Ground Control handoff;
8. one proposed pilot with measurable acceptance.

A D1 demo may use a disclosed manual evidence-packet assembly path.

### Level D2 — customer-operated paid pilot

Requires applicable P0 contract, evidence, access, recovery, source ownership, inference ownership, export, representative evaluation, installation, backup/restore, and support documentation. Customer operates the scoped workflow; a named person reviews every released packet.

### Level D3 — repeatable supported product

Requires at least three substantially similar installations, a supported configuration matrix, repeatable provisioning, measured support/economics, upgrade/rollback evidence, and a release/support policy. This is not currently established.

## Demo acceptance script

Record one uncut run that:

1. displays the paired release manifest and actual host/model configuration;
2. creates or selects one inspection packet;
3. captures a defined Scout image;
4. stores and hashes the untouched original;
5. produces candidate boxes/masks and, where applicable, temporal evidence;
6. shows uncertainty rather than converting empty/unclear output into approval;
7. records an attributed human decision;
8. exports a packet and verifies original hashes and annotation alignment;
9. demonstrates disconnect/reconnect and unavailable evidence behavior;
10. finishes with the exact customer question the pilot will test.

## Metrics

### Software invariants — zero tolerated violations

- wrong-source or wrong-epoch evidence;
- mismatched annotation/original;
- duplicate durable mutation;
- unauthorized control or review mutation;
- stale/unavailable observation shown as current;
- falsely completed business action;
- competing model-load ownership.

### Model/task measurements

- missed target findings;
- false candidates;
- unsupported positive conclusions;
- unresolved rate;
- reviewer acceptance, correction, and rejection counts;
- cold/warm p50/p95 by stage with sample counts;
- memory, model invocations, dropped frames, and failure/degradation behavior.

### Product/economic measurements

- accepted packet completion rate;
- total operator/reviewer minutes versus existing process;
- correction and revisit effort;
- time to first accepted packet;
- installation and support hours;
- direct delivery cost and contribution per deployment;
- paid pilot, renewal, and repeat-install evidence.

`pilot_eval.py` is a useful foundation, but sampled media-timeline detection offset is not customer workflow latency. Add capture, receive, queue, decode, inference, publish, display, review, and export timings where relevant.

## Git and paired-release protocol

The current trees are dirty. At the time this document was created:

- VisionBrain HEAD: `c226a5d81d60`
- visionBrain-bridge HEAD: `24f7e9a579b1`

These hashes do not identify current behavior because modified and untracked work exists.

Before implementation:

1. Inventory every modified and untracked path; assign an owner and disposition.
2. Preserve the current state with reviewed, non-destructive commits on explicitly named productization branches. Do not mix unrelated cleanup.
3. Suggested branch names: `productization/scout-first-v1` in each repository.
4. Record a paired baseline manifest containing both full commit SHAs, dirty-state disposition, Python/Android/toolchain versions, dependency sources, model revisions, supported hardware, configuration, and environment patches.
5. Replace moving-main dependency installation with a tested VisionBrain revision or release.
6. Require every cross-repo protocol PR to link its companion PR and the same fixture revision.
7. Merge contract changes before dependent implementations; do not merge half a wire change.
8. Tag releases with a shared release ID, for example `vb-scout-pilot-v0.1.0`, while retaining independent repository versions.
9. Store acceptance logs, manifests, hashes, and demo references as release artifacts.
10. Never auto-deploy/restart the M2 from CI. Installation remains an explicit authorized step with rollback.

Suggested commit sequence after owner review:

```text
docs: ratify Scout-first productization scope
test: freeze observation and evidence contract fixtures
feat: add durable evidence and review records
feat: enforce inference admission ownership
feat: add Ground Control packet review and export
test: add integrated Scout evidence acceptance
docs: add paired release and operating manifest
```

Do not commit or tag merely because a document exists. CM-00 records the decision and working-tree disposition first.

## This week's team dispatch

### Integration lead

- Ratify VB-P0-01/CM-00.
- Select one demonstration workflow from actual prospect access.
- Record scope, exclusions, reviewer, dataset, supported configuration, and gates.
- Own the paired release manifest and cross-repo decision log.

### Contract lead

- Inventory the existing protocol dialects and current shims.
- Propose the smallest MLX-free released contract subset.
- Produce golden Python fixtures and hand them to bridge/Kotlin/browser owners.

### Core evidence/persistence lead

- Design immutable-original and derived-artifact storage using CM-03 semantics.
- Add review/export records without implementing general business actions.

### Runtime lead

- Reproduce cross-process model contention.
- Design and test the CM-04 host-wide admission contract.

### Ground Control lead

- Prototype the minimum evidence review/export flow against fixtures.
- Do not begin a full UI redesign.

### Evaluation lead

- Define the D1 demo dataset and difficult cases.
- Extend evaluation outputs to record provenance, uncertainty, review, and real timings.

## Explicitly deferred

- repository merger;
- a third backend service;
- full feature parity between both live engines;
- simultaneous multi-camera or multi-tenant service;
- cross-camera identity;
- automatic aircraft/robot motion;
- public-safety or rescue claims;
- production business writes before CM-07;
- cloud/VPS migration;
- custom training, RF-DETR, or Roboflow services without a measured need;
- unrestricted agents;
- broad UI redesign;
- publishing general accuracy, offline, privacy, or performance claims from one demo.

## Release rule

An assisted demonstration is ready only when D1 evidence exists. A customer-operated pilot is ready only when D2's applicable P0 gates pass. The Connected Mission product is ready only under its existing CM/A acceptance rule.

No checklist box, code review, synthetic test, architecture diagram, or founder decision upgrades a capability to customer-ready without its required recorded evidence.
