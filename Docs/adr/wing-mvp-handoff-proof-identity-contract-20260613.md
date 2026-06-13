# ADR: WING MVP Handoff Proof Identity Contract

## Context

The expected WING read-only preflight plan declares the proof artifact schema,
runner, selected MVP blocker, and blocker status. The handoff packet already
declares read-only boundaries, allowed requests, response types, fail-closed
conditions, and blocker anchor metadata, but it does not directly declare this
proof identity.

## Options Considered

1. Keep proof identity only in the expected plan.
2. Add proof identity fields to the handoff packet and validate them against
   the plan and shared contract.
3. Replace plan and packet validation with a larger shared schema object.

## Decision

Add proof identity fields to the handoff packet and make no-WING validation
require them to match both the shared WING preflight contract and expected plan
artifact.

## Why This Won

It removes a handoff drift point while preserving the existing artifact shapes
and read-only safety boundary.

## Rejected Alternatives

Plan-only identity keeps packet review weaker. A full schema object may be
useful later, but it is broader than required for the current blocker-first MVP
rail.

## Implementation Plan

Expose the four proof identity fields from the packet builder, render them in
markdown, validate them in `tools/no_wing_release_check.py`, surface them in
summary output, and add focused positive and negative tests.

## Test Plan

Run focused handoff/no-WING tests, no-WING quick, project rail tests, direction
guard, memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to contract fixture mismatch. Roll back the packet fields and
matching validation if the identity binding is wrong. No live-write capability
or readiness closure is introduced.
