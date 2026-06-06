# Proposal: operator-proposal-apply-sequence-dry-run-20260606

## Short Thesis

Add `dry_run_operator_proposal_apply_sequence` as the final no-write rehearsal
before any future supervised sequence executor.

## What I Understood

The project needs to move from recommendations toward autonomous execution, but
the live path must remain auditable. A dry-run rehearsal lets the system verify
that a plan is executable without mutating idempotency, queue, or console state.

## Proposed Solution

- Accept either a full `sequence_plan` payload or a direct
  `planned_apply_sequence` list.
- Recheck every step against the current `OperatorProposalQueue`.
- Verify plan safety flags, `apply_operator_proposal` request shape, unique
  idempotency keys, and fader/gain-only normalized changes.
- Return per-step `would_apply` or blocked rows with explicit no-write flags.
- Advertise the dry-run handler from the Decision Queue apply sequence summary.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-dry-run-20260606.md`

## Alternatives Considered

- Implement a live sequence executor now. Rejected until dry-run evidence
  exists for plan validation and stale-state handling.
- Re-run preflight from the dry-run endpoint. Rejected because preflight and
  execution rehearsal should remain separate evidence stages.
- Trust plan steps directly. Rejected because plans can be stale or forged.

## Risks

Dry-run output can be mistaken for permission to execute. Mitigation: the
payload uses `execution_allowed_by_dry_run=false`, `dry_run_only=true`, and
`does_not_call_apply_operator_proposal=true`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_dry_run_operator_proposal_apply_sequence_reports_would_apply_without_write -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_dry_run_operator_proposal_apply_sequence_blocks_stale_plan_without_write -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_dry_run_operator_proposal_apply_sequence_blocks_forged_idempotency_mismatch -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer moving directly to an executor. I would keep this dry-run stage
because it turns the executor's future preconditions into testable evidence.
