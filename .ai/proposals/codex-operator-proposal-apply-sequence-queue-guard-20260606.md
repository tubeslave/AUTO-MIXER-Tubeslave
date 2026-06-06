# Proposal: operator-proposal-apply-sequence-queue-guard-20260606

## Short Thesis

Harden `plan_operator_proposal_apply_sequence` by making current pending queue
state mandatory for every planned apply step.

## What I Understood

Preflight readiness is time-sensitive. A result that was valid a moment ago can
be unsafe after queue state changes, so the planner needs a stale-result guard
before any future sequence executor can rely on its output.

## Proposed Solution

- Look up each ready result's `proposal_id` in `server.operator_proposal_queue`.
- Require the current proposal to be `pending`, not expired, and still
  `can_apply=true`.
- Compare preflight `updated_at`, `requested_change`, and `requested_changes`
  against the current proposal snapshot when the preflight payload provides
  those fields.
- Include the current proposal snapshot in each planned step.
- Block missing, non-pending, expired, no-longer-applyable, or changed results
  without mutating the queue or applying anything.

## Likely Files To Touch

- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-queue-guard-20260606.md`

## Alternatives Considered

- Trust preflight results exactly as supplied. Rejected because preflight
  readiness can go stale.
- Re-run batch preflight from the planner. Rejected because planning should not
  hide pre-read queries.
- Move directly to a sequence executor. Rejected until the stale-result guard is
  explicit and tested.

## Risks

The guard may block externally supplied preflight results that are not tied to
the current queue. That is intentional for live-sound safety: apply planning is
only valid for current queue proposals.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_builds_read_only_plan_from_batch_preflight -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_rejects_forged_ready_result -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_blocks_stale_non_pending_result -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi may argue that preflight results should be portable evidence. I would keep
apply-sequence planning queue-local; portable evidence belongs in reports, not
in a live-apply preparation path.
