# ADR: WING Read-Only Runner Stale Prestart Plan Regression

## Context

The WING read-only preflight runner requires an offline expected plan before it
opens a backend WebSocket. The backend-port prestart failure-detail contract now
includes `listener_confirmation_checklist`; a stale plan missing that field
must fail closed before network contact.

## Options considered

1. Rely on generic expected-plan mismatch coverage.
2. Add a specific stale-prestart-detail regression at runner level.
3. Add new production validation code.

## Decision

Use option 2. The runner test now mutates a valid expected plan to remove
`listener_confirmation_checklist` and proves the preflight blocks before
WebSocket connect.

## Why this won

It directly protects the first WING MVP blocker without changing runtime
behavior or widening authority. The runner was already capable of rejecting the
mismatch; the evidence needed to be explicit.

## Rejected alternatives

Generic mismatch coverage was too broad to prove this specific operator gate.
New production code was unnecessary because the existing match-key mechanism
already covers the field.

## Implementation plan

- Build a valid expected plan in the test.
- Remove the listener checklist from prestart failure-detail fields.
- Assert `expected_plan_mismatch`, no sent requests, and no WebSocket open.

## Test plan

- Focused runner expected-plan tests.
- Adjacent WING/no-WING tests.
- Full repo validation before commit.

## Risks and rollback

The risk is limited to test maintenance if the shared contract changes again.
Rollback by updating the test mutation to the new required field.
