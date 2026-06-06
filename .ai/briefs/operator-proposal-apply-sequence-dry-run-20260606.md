# Brief: operator-proposal-apply-sequence-dry-run-20260606

## Context

The Product Layer can build a queue-fresh apply sequence plan from batch
preflight results. Before a real supervised sequence executor exists, operators
need a dry-run rehearsal that validates the plan as an execution artifact
without applying it.

## Objective

Add a dry-run handler that accepts a planned apply sequence, rechecks current
queue and idempotency evidence, and reports `would_apply` or blocked rows.

## Safety Constraints

- Do not call `apply_operator_proposal`.
- Do not call `_apply_manual_console_write`.
- Do not reserve idempotency keys.
- Do not mutate proposal status, queue state, or delta history.
- Keep live write kinds limited to the existing fader/gain surface.
- Treat dry-run output as evidence only, not execution authority.

## Expected Evidence

- Valid planned steps become `would_apply` dry-run rows.
- Stale, forged, missing-idempotency, duplicate-idempotency, or unsupported
  planned steps are blocked.
- Queue state, idempotency storage, cumulative-delta history, and mixer writes
  remain unchanged.
