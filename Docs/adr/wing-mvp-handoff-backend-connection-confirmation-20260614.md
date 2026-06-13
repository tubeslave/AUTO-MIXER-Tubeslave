# ADR: WING handoff backend connection confirmation

## Context

The WING read-only handoff is the operator-visible bridge from no-WING local
checks to a real read-only hardware capture. It already records that physical
preflight requires an existing backend connection, but the operator sequence
also needs a validated confirmation step before capture.

## Options considered

1. Keep the confirmation only as `physical_preflight_requires_existing_backend_connection`.
2. Add a dedicated operator-sequence confirmation step before capture.
3. Defer the check to the real preflight runner only.

## Decision

Add a shared backend connection confirmation step and require no-WING handoff
validation to prove the step exists at index `2` before
`Run capture_readonly_preflight.`.

## Why this won

The real WING run is safety-sensitive. The handoff packet must make the
operator confirm the existing backend connection is the intended WING Rack
before the capture command can be followed.

## Rejected alternatives

Keeping only the boolean boundary was rejected because it does not prove
operator-sequence ordering. Deferring entirely to the real runner was rejected
because stale handoff packets should be rejected before the hardware session.

## Implementation plan

- Define the shared confirmation step and required markers.
- Use the shared step in handoff generation.
- Add no-WING validation fields and violations for missing or late steps.
- Update Project Director and WING MVP rail text.

## Test plan

- Focused handoff/no-WING tests.
- no-WING quick release check.
- Project Director/WING rail tests.
- Full pytest before commit.

## Risks and rollback

Older handoff packets fail until regenerated. Rollback is to remove the new
step contract and summary fields, but that weakens the handoff ordering before
real WING capture.
