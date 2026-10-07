# Expertise-only Watch / Jev amendment

Date: 2026-10-07. Authorized by the user's explicit instruction to implement
the bridge [EXPERTISE_AUTOTARGET_JEV_SPEC.md](../../visionBrain-bridge/docs/EXPERTISE_AUTOTARGET_JEV_SPEC.md).

This scoped amendment authorizes deliberate changes to the canonical mission
contracts, model adapters and runtime for expertise-only Watch. An inferred
objective is selected from actual-frame vision proposals, kept separate from
operator-authored `goal`, grounded through SAM and committed with the existing
bridge arbiter's acknowledged target revision. Legacy Inspect/Watch semantics
remain available when the new extension is not requested.

The optional inferred mode may call TypeSafe Jev through OpenRouter, using
protected host credentials and context/derived candidate text only. This is the
narrow exception to the earlier no-runtime-network/no-new-cloud-dependency
guidance in AGENTS.md, NEXT_DEVELOPMENT_SPEC.md and CM-00. It authorizes no
model downloads, unrelated models/providers, raw-frame export, or second mission authority.

CM-00 evidence, provenance, serialized inference, source ownership, bounded
execution, manual override and release constraints remain binding. Approval
does not qualify a model, restart the production hub, install a phone build,
push, merge or release. Actual implementation/acceptance status is recorded in
the bridge [implementation record](../../visionBrain-bridge/docs/EXPERTISE_AUTOTARGET_JEV_IMPLEMENTATION.md).

## Provider amendment — October 7

The user explicitly requested `https://openrouter.ai/typesafe/jev-1.13` and
supplied an OpenRouter credential. The authorized transport is now
`POST https://openrouter.ai/api/alpha/decisions`, with request model
`typesafe/jev-1.13` and protected host `OPENROUTER_API_KEY`. OpenRouter
routes the decision to TypeSafe; the local vision/SAM path is unchanged.
The documented resolved model is `typesafe/jev-1.13-20260917`; record the actual
response model and reject unexpected versions. No direct TypeSafe key is required.

[OpenRouter Jev documentation](https://openrouter.ai/docs/guides/community/jev)
and [Decisions API reference](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request).
