# ADR: operator-proposal-apply-sequence-dry-run-20260606

## Context

The system can plan supervised apply sequences but does not yet have a safe
execution rehearsal between planning and future live execution.

## Options Considered

1. Add a real sequence executor immediately.
2. Add a dry-run execution rehearsal first.
3. Keep sequence planning as the final step.

## Decision

Use option 2. Add `dry_run_operator_proposal_apply_sequence` as a no-write
handler that validates planned steps and returns `would_apply` or blocked rows.

## Why This Won

It moves the system toward supervised automation while keeping execution,
idempotency reservation, queue mutation, and console writes out of the dry-run
contract.

## Rejected Alternatives

- A real executor was deferred because it would add a live-write surface before
  the plan-as-input contract has no-write evidence.
- Stopping at planning was rejected because the future executor needs a stable
  rehearsal payload and tests.
- Hidden preflight was rejected because the dry-run should not query pre-read
  state or mutate cumulative delta history.

## Implementation Plan

1. Add the dry-run handler and dispatch registration.
2. Accept full sequence plans or direct planned-step lists.
3. Recheck current queue proposal state and idempotency keys.
4. Validate apply request shape and fader/gain-only planned changes.
5. Return no-write safety flags and per-step dry-run rows.
6. Add focused tests for valid and stale plans.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_dry_run_operator_proposal_apply_sequence_reports_would_apply_without_write -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_dry_run_operator_proposal_apply_sequence_blocks_stale_plan_without_write -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_dry_run_operator_proposal_apply_sequence_blocks_forged_idempotency_mismatch -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: dry-run rows may be treated as execution authority. Mitigation: the
payload states `execution_allowed_by_dry_run=false`, `dry_run_only=true`, and
`does_not_call_apply_operator_proposal=true`.

Rollback: remove the dry-run handler and Decision Queue discovery fields. The
sequence planner remains unchanged.
