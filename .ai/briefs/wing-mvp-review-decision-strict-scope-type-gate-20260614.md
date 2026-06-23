# Brief: WING MVP Review Decision Strict Scope Type Gate

## Task ID

wing-mvp-review-decision-strict-scope-type-gate-20260614

## Context

The WING read-only evidence and review-template layers reject coerced scope and
count values, but the review decision ledger still parsed manual decision
record channel lists with `int(...)`. That allowed strings or floats to match a
candidate's integer channel scope.

## Goal

Require manual review decision `channels` and `operator_confirmed_channels`
values to be real integers before a decision record can be accepted.

## Safety Boundary

This is offline/read-only ledger validation. It must not contact WING, grant
live authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
