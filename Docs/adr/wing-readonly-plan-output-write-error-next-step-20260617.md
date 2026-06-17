# ADR: WING Read-Only Plan Output Write Error Next Step

## Context

The WING MVP remains blocked on real read-only connection proof. Before any
backend/WING WebSocket is opened, the runner can emit a plan-only expected-plan
artifact. If `--plan-output` points to an unwritable path, the runner already
fails closed, denies readiness, and reports the write error. The blocked output
did not tell the operator how to recover safely.

## Options considered

1. Keep the blocked write-error output as a machine-readable failure only.
2. Add an `operator_next_step` to the write-error blocked plan.
3. Raise the write error and rely on stderr or a traceback.

## Decision

Use option 2. When plan-output writing fails, the blocked plan now tells the
operator to choose a writable `--plan-output` path, rerun with `--plan-only`,
and do that before opening the backend/WING WebSocket.

## Why this won

The write-error path is a recovery gate before real WING contact. Clear operator
guidance keeps the next action inside the offline evidence chain and avoids
guesswork that could bypass the expected-plan artifact.

## Rejected alternatives

- Option 1 preserves fail-closed behavior but leaves the operator without a
  concrete recovery action.
- Option 3 is less useful for automated summaries and can obscure the safety
  state that readiness and live authority remain denied.

## Implementation plan

- Add `operator_next_step` to `block_plan_output_write_error`.
- Extend the existing plan-output write-error regression to assert the recovery
  guidance and the no-WebSocket boundary.

## Test plan

- Targeted plan-output write-error regression.
- no-WING quick release check.
- Focused WING preflight, no-WING, handoff, and Project Director tests.
- `git diff --check`.
- Full pytest.

## Risks and rollback

The risk is limited to a blocked plan-only error surface. Rollback is to remove
the `operator_next_step` field and its assertion; the existing readiness denial
would remain intact.
