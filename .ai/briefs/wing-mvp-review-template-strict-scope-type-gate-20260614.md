# Brief: WING MVP Review Template Strict Scope Type Gate

## Task ID

wing-mvp-review-template-strict-scope-type-gate-20260614

## Context

Evidence intake and review-decision ledger paths now reject malformed channel
scope values, but the review decision template builder also accepts already
built report mappings. A hand-built or stale report object could still carry
string or float channel lists into template generation.

## Goal

Make the review decision template fail closed unless `channels` and
`operator_confirmed_channels` are lists of real integers.

## Safety Boundary

This is offline/read-only review tooling. It must not contact WING, grant live
authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
