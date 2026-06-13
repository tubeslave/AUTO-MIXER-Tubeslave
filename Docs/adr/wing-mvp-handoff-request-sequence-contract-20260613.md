# ADR: WING MVP Handoff Request Sequence Contract

## Context

The WING read-only preflight plan exposes the exact request sequence sent
during the real run. The handoff packet asked the operator to confirm that
sequence, but did not expose the sequence itself.

## Options Considered

1. Leave request sequence only in the expected plan.
2. Add request sequence to the handoff packet and validate packet/plan
   agreement.
3. Replace the handoff packet and expected plan with a larger shared schema.

## Decision

Add `request_sequence` and `request_count` to the handoff packet, render them
in markdown, and fail no-WING validation when the packet differs from the
generated plan or expected sequence.

## Why This Won

It improves operator review of the real WING step and catches stale handoff
packets without changing live command paths or granting authority.

## Rejected Alternatives

Plan-only request sequence was rejected because the handoff packet is the
operator's primary hardware-step artifact. A broader schema migration was
rejected as too large for this narrow readiness hardening step.

## Implementation Plan

Build the handoff request sequence from the packet channel and value-type
scope, render it, validate it in `tools/no_wing_release_check.py`, and cover
happy-path plus drift tests.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, project
direction guards, memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to stale handoff packets failing closed. Roll back by reverting
this patch; no live-write path is introduced.
