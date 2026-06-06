# Brief: operator-proposal-apply-sequence-executor-20260606

## Context

The Product Layer can now build a read-only apply sequence plan, rehearse it as
a dry run, and gate dry-run rows into `ready_for_executor` candidates after
operator acknowledgement. The project still lacks the first real sequence
executor for supervised fader/gain proposals.

## Objective

Add `execute_operator_proposal_apply_sequence` as a tightly gated executor that
consumes execution-gate output and delegates each ready row to the existing
`apply_operator_proposal` handler.

## Safety Constraints

- Require `execute=true`, `operator_ack=true`, and `approved=true`.
- Require current operator mode to allow supervised console apply.
- Accept only `ready_for_executor` rows from the execution gate shape.
- Recheck current queue proposal state, apply request, idempotency uniqueness,
  and fader/gain-only normalized changes before attempting any apply.
- Do not call `_apply_manual_console_write` directly.
- Do not reserve idempotency keys directly; reservation remains inside
  `apply_operator_proposal`.
- Stop before applying if any input row fails validation.
- Stop on the first delegated apply that does not produce an applied proposal.

## Expected Evidence

- Missing execution acknowledgement blocks without writes or idempotency
  reservation.
- A valid gate row is applied through `apply_operator_proposal` and moves the
  proposal to applied history.
- Stale gate rows block before any delegated apply.
- Decision Queue visibility names the executor while preserving the current
  fader/gain live-write boundary.
