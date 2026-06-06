# Proposal: operator-proposal-apply-sequence-execution-gate-20260606

## Short Thesis

Add `gate_operator_proposal_apply_sequence_execution` as a read-only validation
gate between dry-run rehearsal and any future supervised sequence executor.

## What I Understood

The project needs to advance toward autonomous execution, but every transition
toward live writes needs auditable evidence. Dry-run rows are useful, but a
future executor should consume a stricter gate output that proves operator ack,
current queue state, idempotency shape, and fader/gain limits.

## Proposed Solution

- Accept `dry_run_payload` or direct `dry_run_results`.
- Require `operator_ack=true`.
- Require current operator mode to allow supervised console apply.
- Accept only `would_apply` dry-run rows.
- Recheck current queue proposal state, apply request shape, idempotency keys,
  and fader/gain-only normalized changes.
- Return `ready_for_executor` rows with `execution_allowed_by_gate=false`.
- Advertise the gate from Decision Queue apply-sequence visibility.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-execution-gate-20260606.md`

## Alternatives Considered

- Implement the real sequence executor now. Rejected until the gate output is
  stable and tested.
- Let the executor consume dry-run rows directly. Rejected because dry-run rows
  should be validated one final time against operator ack and current queue
  state.
- Fold the gate into dry-run. Rejected because dry-run and operator-ack gating
  are distinct evidence stages.

## Risks

Gate output can be mistaken for permission to execute. Mitigation: payloads use
`execution_allowed_by_gate=false`, `requires_separate_executor=true`, and
`does_not_call_apply_operator_proposal=true`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_execution_requires_operator_ack -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_execution_reports_ready_without_write -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_gate_operator_proposal_apply_sequence_execution_blocks_stale_dry_run_without_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer collapsing gate and executor. I would keep them separate because
the gate gives future executor work a small, testable precondition contract.
