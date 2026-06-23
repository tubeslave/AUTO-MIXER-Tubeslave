# Brief: WING MVP Evidence Strict Integer Scope Gate

## Task ID

wing-mvp-evidence-strict-integer-scope-gate-20260614

## Context

The WING read-only preflight evidence validator normalizes declared channel
scope before deciding whether a capture can become a manual-review candidate.
After rejecting booleans, the helper still accepted numeric strings and floats
through `int(...)` coercion.

## Goal

Require declared WING channel scope values to be real JSON integers before a
read-only preflight artifact can be accepted.

## Safety Boundary

This is offline/read-only artifact validation. It does not contact WING, grant
live authority, mutate readiness, or widen supervised write permissions.

## Test Plan

- Focused WING evidence/review/no-WING tests.
- Standard repository pytest before any commit attempt.
