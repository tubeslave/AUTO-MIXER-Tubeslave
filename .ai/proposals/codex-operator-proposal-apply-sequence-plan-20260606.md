# Proposal: operator-proposal-apply-sequence-plan-20260606

## Short Thesis

Add `plan_operator_proposal_apply_sequence` as a read-only bridge from batch
preflight evidence to an ordered supervised apply plan.

## What I Understood

Batch preflight proves which pending proposals are currently ready or blocked,
but it does not tell the operator how to apply the ready set. The next autonomy
step is an execution plan contract, not automatic execution.

## Proposed Solution

- Add a Product Layer handler that accepts `preflight_results`.
- Reject missing or malformed results instead of running hidden preflight.
- For each ready result, verify the no-live-write safety envelope,
  idempotency evidence, and normalized fader/gain changes.
- Return ordered plan steps containing `apply_operator_proposal` request
  payloads, normalized change evidence, rollback source values, and explicit
  `execution_allowed_by_plan=false`.
- Add Decision Queue visibility for the planner so the UI can discover the next
  operator action after batch preflight.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-plan-20260606.md`

## Alternatives Considered

- Have the planner run batch preflight internally. Rejected because this would
  hide pre-read checks behind a planning endpoint.
- Automatically apply all ready results. Rejected because preflight readiness
  is not operator approval and not write/readback/rollback proof.
- Keep planning only in the frontend. Rejected because the safety envelope
  should be server-side and testable.

## Risks

The plan could be mistaken for authority to execute. Mitigation: every payload
marks `execution_allowed_by_plan=false`, `requires_operator_apply=true`, and
points to the existing `apply_operator_proposal` gate.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_builds_read_only_plan_from_batch_preflight -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_rejects_forged_ready_result -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may prefer jumping directly to a batch executor. I would keep this as a
separate evidence contract so the executor can later be small, auditable, and
forced back through the existing supervised apply path.
