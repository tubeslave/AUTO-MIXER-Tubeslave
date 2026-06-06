# ADR: operator-proposal-apply-sequence-execution-gate-20260606

## Context

Dry-run apply sequence rehearsal can say which planned steps would apply, but
there is no final read-only gate for operator acknowledgement and current queue
state before a future sequence executor.

## Options Considered

1. Add a real sequence executor immediately.
2. Let the executor consume dry-run rows directly.
3. Add a read-only execution gate first.

## Decision

Use option 3. Add `gate_operator_proposal_apply_sequence_execution` to validate
dry-run rows, operator acknowledgement, and current queue state without applying
anything.

## Why This Won

It advances the supervised automation pipeline while preserving the existing
live-write boundary. A future executor can consume the gate output instead of
raw dry-run rows.

## Rejected Alternatives

- Immediate execution was rejected because live writes need a smaller, proven
  precondition contract.
- Consuming dry-run directly was rejected because dry-run rows can go stale or
  be forged.
- Combining dry-run and gate was rejected because operator acknowledgement is a
  separate safety event.

## Implementation Plan

1. Add gate handler and dispatch registration.
2. Accept full dry-run payloads or direct result lists.
3. Require `operator_ack=true` and supervised apply capability.
4. Recheck `would_apply` rows against current queue, apply request, idempotency,
   and planned fader/gain changes.
5. Return `ready_for_executor` or blocked rows with explicit no-write flags.
6. Add Decision Queue discovery fields.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_execution_requires_operator_ack -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_execution_reports_ready_without_write -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_execution_blocks_stale_dry_run_without_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: gate output may be confused with execution permission. Mitigation:
`execution_allowed_by_gate=false`, `requires_separate_executor=true`, and
`live_write=false` are included in summary, safety, and candidate rows.

Rollback: remove the gate handler and Decision Queue discovery fields. Sequence
planning and dry-run rehearsal remain available.
