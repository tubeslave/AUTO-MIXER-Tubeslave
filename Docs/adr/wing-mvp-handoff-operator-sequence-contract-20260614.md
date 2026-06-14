# ADR: WING MVP Handoff Operator Sequence Contract

## Context

The supervised WING MVP waits on real WING read-only connection proof. The
handoff packet now declares a SHA256 digest over its operator sequence, but that
digest only proves the packet has not changed after generation. The exact
ordered checklist still needed a shared source of truth.

## Options considered

1. Keep builder-local sequence construction plus SHA validation.
2. Duplicate an expected sequence in the no-WING validator.
3. Move the expected sequence into the shared preflight contract and have both
   builder and validator use it.

## Decision

Use option 3.

## Why this won

A shared contract prevents builder-local checklist drift and lets no-WING
validation reject a regenerated packet whose sequence no longer matches the
canonical supervised WING handoff order. It remains offline and read-only.

## Rejected alternatives

Builder-local SHA validation misses generated drift. Validator-local duplicate
lists create a second source of truth and raise maintenance risk.

## Implementation plan

Add `expected_handoff_operator_sequence()` and step constants to the shared
contract. Have the handoff builder emit expected sequence SHA and match flags.
Have no-WING validation recompute the expected sequence, expose match fields,
and fail closed on contract mismatch. Update docs and tests.

## Test plan

Run focused no-WING and handoff packet tests, WING MVP rails tests, no-WING
quick, Project Director, direction guard, memory lint, `git diff --check`, and
the standard full test command.

## Risks and rollback

Risk is limited to stricter offline handoff validation. Rollback is to remove
the shared sequence contract fields and mismatch violations if the packet schema
changes.
