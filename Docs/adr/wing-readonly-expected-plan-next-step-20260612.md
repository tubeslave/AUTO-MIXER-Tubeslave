# ADR: WING Read-Only Expected Plan Next Step

## Context

The WING read-only preflight runner fails closed before WebSocket contact when
the expected offline plan is missing, invalid, or mismatched. The generated
summary did not include `operator_next_step` for those cases.

## Options considered

1. Keep guidance only in the runbook.
2. Add `operator_next_step` to blocked expected-plan summaries.
3. Print console-only instructions without changing summary artifacts.

## Decision

Use option 2. The durable `summary.json` now carries actionable next-step text
for expected-plan missing, invalid, and mismatch branches.

## Why this won

The first WING MVP hardware step is operator-driven. When the gate blocks before
WebSocket, the artifact should say how to recover without guessing or touching
live state.

## Rejected alternatives

Runbook-only guidance was rejected because blocked artifacts may be reviewed
out of context. Console-only guidance was rejected because it is not durable
evidence.

## Implementation plan

- Add an expected-plan next-step helper.
- Populate `operator_next_step` in missing, invalid, and mismatch summaries.
- Extend runner tests.
- Mention the field in the runbook.

## Test plan

- Runner expected-plan tests.
- Full runner test file.
- Full repo validation before commit.

## Risks and rollback

The main risk is wording drift. Roll back by removing the helper and assertions;
the underlying fail-closed behavior remains intact.
