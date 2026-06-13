# ADR: WING MVP Handoff Expected Artifacts Contract

## Context

The WING read-only preflight expected plan declares two expected artifacts for
a real run: summary JSON and event log JSONL. The handoff packet already
declares named summary/event globs, but it did not carry the same list shape as
the expected plan or validate that the packet list matched the plan.

## Options Considered

1. Keep only the named handoff glob fields.
2. Add a handoff `expected_artifacts` list and validate it against the expected
   plan and named fields.
3. Replace the handoff artifact section with a larger schema model.

## Decision

Add `expected_artifacts` to the handoff packet and make no-WING validation
require it to match both the shared expected artifact globs and the expected
plan artifact list.

## Why This Won

It removes a plan-to-packet drift point while preserving the current packet
shape and read-only safety boundary.

## Rejected Alternatives

Named fields alone are less audit-friendly. A larger schema model may be useful
later, but is broader than required for this blocker-first step.

## Implementation Plan

Expose the expected artifact list from the packet builder, validate list and
named summary/event globs in `tools/no_wing_release_check.py`, surface the
comparison in summary output, and add positive/negative tests.

## Test Plan

Run focused handoff/no-WING tests, no-WING quick, project rail tests, direction
guard, memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to artifact path fixture mismatch. Roll back the packet list
and matching validation if the binding is wrong. No live-write capability or
readiness closure is introduced.
