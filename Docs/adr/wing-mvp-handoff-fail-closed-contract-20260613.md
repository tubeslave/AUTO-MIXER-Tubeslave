# ADR: WING MVP Handoff Fail-Closed Contract

## Context

The WING read-only preflight plan exposes the fail-closed conditions that must
stop a real run safely. The operator handoff packet did not expose or validate
that same list.

## Options Considered

1. Leave fail-closed conditions only in the expected plan.
2. Add fail-closed conditions to the handoff packet and validate packet/plan
   agreement.
3. Replace the handoff packet and expected plan with a larger shared schema.

## Decision

Add `fail_closed_conditions` to the handoff packet, render it in markdown, and
fail no-WING validation when the packet differs from the shared contract or
expected plan artifact.

## Why This Won

It improves operator review of the real WING step and catches stale handoff
packets without changing any live command path or granting authority.

## Rejected Alternatives

Plan-only fail-closed conditions were rejected because the handoff packet is the
operator's primary hardware-step artifact. A broader schema migration was
rejected as too large for this focused readiness hardening step.

## Implementation Plan

Derive the handoff list from
`WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS`, render it, validate it
in `tools/no_wing_release_check.py`, and cover happy-path plus drift tests.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, project
direction guards, memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to stale handoff packets failing closed. Roll back by reverting
this patch; no live-write path is introduced.
