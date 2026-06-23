# Brief: WING MVP Review Template Value Type Gate

## Task ID

wing-mvp-review-template-value-type-gate-20260614

## Context

The WING read-only review decision template builder now rejects malformed
channel scope and count fields, but it still treated any non-empty
`value_types` list as candidate identity. A hand-built or stale report object
could carry unsupported values such as `eq` or non-string entries into pending
decision template generation.

## Goal

Make the review decision template fail closed unless `value_types` is a
duplicate-free list of supported WING read-only preflight value-type strings.

## Safety Boundary

This is offline/read-only review tooling. It must not contact WING, grant live
authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
