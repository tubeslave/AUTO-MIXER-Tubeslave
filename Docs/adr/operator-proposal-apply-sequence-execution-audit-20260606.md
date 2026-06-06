# ADR: operator-proposal-apply-sequence-execution-audit-20260606

## Context

The sequence executor can now apply ready rows through `apply_operator_proposal`.
That is useful for Supervised AutoMix, but future autonomy needs an explicit
post-execution proof gate that rejects forged direct-write claims and missing
readback evidence.

## Options Considered

1. Trust the executor summary directly.
2. Add a read-only execution audit handler.
3. Persist execution audit records immediately.

## Decision

Use option 2. Add `audit_operator_proposal_apply_sequence_execution` as a
read-only handler that validates executor result payloads and emits pass/block
evidence without mutating runtime state.

## Why This Won

It creates a proof layer between supervised execution and any future autonomous
policy consumer. The audit can validate delegated apply path, idempotency,
allowed write kinds, readback evidence, and no direct-write claims without
adding another live or durable write path.

## Rejected Alternatives

- Trusting the executor summary was rejected because summary flags can be
  forged by callers.
- Immediate persistence was rejected because it adds a separate write contract.
- Re-running apply gates was rejected because the audit should inspect the
  result evidence, not perform another apply-stage operation.

## Implementation Plan

1. Add audit request parsing for `execution_result_payload` and direct rows.
2. Validate executor summary and safety envelope.
3. Validate each applied execution row and its apply result.
4. Emit read-only audit rows, blockers, and pass/block summary.
5. Advertise the audit handler in Decision Queue apply-sequence visibility.
6. Update durable project memory with the audit-only boundary.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_audit_operator_proposal_apply_sequence_execution_passes_verified_executor_result -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_audit_operator_proposal_apply_sequence_execution_blocks_forged_direct_write_claim -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: audit pass may be confused with execution authority. Mitigation:
`audit_only=true`, `execution_allowed_by_audit=false`, `live_authority_granted=false`,
and `does_not_call_apply_operator_proposal=true` are part of the contract.

Rollback: remove the audit handler and Decision Queue discovery fields. The
executor and earlier plan/dry-run/gate stages remain available.
