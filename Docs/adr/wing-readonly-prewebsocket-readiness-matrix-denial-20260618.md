# WING Read-Only Pre-WebSocket Readiness Matrix Denial

## Context

The supervised WING MVP still depends on a real read-only connection proof before any readiness closure or matrix update. The preflight runner already denied live authority and readiness closure on offline and pre-WebSocket surfaces, but those surfaces did not all make the readiness-matrix update denial explicit.

## Options considered

1. Leave readiness-matrix denial implicit through `readiness_closure_allowed: false`.
2. Add a top-level `readiness_matrix_update_allowed: false` field to offline and pre-WebSocket runner outputs.

## Decision

Add `readiness_matrix_update_allowed: false` to the offline plan, blocked pre-WebSocket plan outputs, pre-WebSocket real-run summary, and validate-inputs-only summary.

## Why this won

The first WING blocker is not closed by a plan, argument validation, or pre-WebSocket artifact. The explicit field keeps downstream review and release checks from mistaking a valid plan for a readiness-matrix update candidate.

## Rejected alternatives

Relying only on `readiness_closure_allowed: false` was rejected because the readiness matrix is a distinct project-tracking surface.

## Implementation plan

Update the runner summary dictionaries and expected-plan match keys, then cover the new denial in the existing no-WebSocket runner tests.

## Test plan

Run targeted runner tests, no-WING quick validation, the focused WING/no-WING suite, `git diff --check`, and the standard full pytest command.

## Risks and rollback

Risk is low because the change only adds a false-valued reporting field before live contact. Rollback by removing the field and its assertions if a downstream artifact schema rejects the additional key.
