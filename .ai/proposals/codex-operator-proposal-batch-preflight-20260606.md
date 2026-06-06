# Proposal: operator-proposal-batch-preflight-20260606

## Short Thesis

Add a dry-run `preflight_operator_proposal_apply_batch` handler so the operator
cockpit can validate the current apply candidates in one request before any
supervised apply.

## What I Understood

The queue now shows candidates and the single-proposal preflight can validate
one proposal. The next useful autonomy step is not broader execution; it is a
batch proof surface that makes the whole pending apply set predictable.

## Proposed Solution

- Add `preflight_operator_proposal_apply_batch` beside the single preflight and
  apply handlers.
- Select pending `can_apply=true` proposals by default, or explicit
  `proposal_ids` when supplied.
- Use the same validation helper used by apply/preflight, with delta-history
  mutation disabled.
- Generate per-proposal dry-run idempotency keys from an operator-supplied
  prefix, or surface missing-key blockers when no prefix/key is provided.
- Return summary counts, result IDs, and explicit no-live-write safety flags.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-batch-preflight-20260606.md`

## Alternatives Considered

- Trigger batch preflight automatically in `build_decision_queue`. Rejected
  because queue snapshots must stay read-only and cheap.
- Route batch preflight through repeated websocket sends from the single
  handler. Rejected because the batch endpoint should return one stable payload.
- Apply all ready proposals automatically. Rejected because apply remains a
  separate supervised action.

## Risks

Batch preflight can drift from single preflight if validation logic forks too
far. Mitigation: reuse the existing normalization and
`validate_change_before_apply` helper and keep the batch result format aligned
with the single preflight payload.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_batch_preflight_operator_proposal_apply_reports_ready_and_blocked_without_live_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Where The Other Agent May Disagree

Kimi may ask to fully extract apply validation first. I would keep this handler
local for now because it is still Product Layer behavior and the tests pin the
no-mutation boundary.
