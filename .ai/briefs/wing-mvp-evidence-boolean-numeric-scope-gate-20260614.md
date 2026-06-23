# Brief: WING MVP Evidence Boolean Numeric Scope Gate

## Task ID

wing-mvp-evidence-boolean-numeric-scope-gate-20260614

## Context

The WING read-only preflight evidence intake normalizes channel ids and readback
counts before deciding whether a captured artifact can become a manual-review
candidate. Python treats booleans as integers, so unchecked helpers can turn
`true` into `1`.

## Goal

Reject boolean values in numeric WING scope and readback-count fields before a
read-only preflight artifact can be accepted.

## Safety Boundary

This is an offline/read-only validation change. It must not contact WING, grant
live authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
