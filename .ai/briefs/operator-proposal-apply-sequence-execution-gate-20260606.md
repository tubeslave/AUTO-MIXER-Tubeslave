# Brief: operator-proposal-apply-sequence-execution-gate-20260606

## Context

The Product Layer can dry-run a planned apply sequence and produce `would_apply`
evidence without mutating the queue or console. The next boundary before a real
executor is a read-only gate that checks operator acknowledgement and current
state one more time.

## Objective

Add a read-only execution gate that accepts dry-run rows, requires explicit
operator acknowledgement, and returns `ready_for_executor` candidates or blocked
rows.

## Safety Constraints

- Do not call `apply_operator_proposal`.
- Do not call `_apply_manual_console_write`.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal status, or delta history.
- Require current pending queue proposals and fader/gain-only planned changes.
- Gate output is not execution authority; a separate executor is still required.

## Expected Evidence

- Missing `operator_ack` blocks the gate.
- Valid dry-run rows become `ready_for_executor` candidates.
- Stale, duplicate-idempotency, forged, or non-`would_apply` rows are blocked.
- Queue state, idempotency storage, and mixer writes remain unchanged.
