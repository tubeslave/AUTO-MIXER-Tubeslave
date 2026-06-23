# Brief: WING MVP Review Template Count Type Gate

## Task ID

wing-mvp-review-template-count-type-gate-20260614

## Context

The WING read-only evidence intake now rejects malformed numeric scope fields,
but the manual review decision template builder still normalized candidate
readback counts with `int(...)`. A hand-built or stale report object could turn
`true` or `"3"` into valid-looking counts.

## Goal

Make the review decision template fail closed unless readback counts are real
integers.

## Safety Boundary

This is offline/read-only review tooling. It must not contact WING, grant live
authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
