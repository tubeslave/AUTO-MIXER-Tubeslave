# ADR: operator-proposal-apply-sequence-queue-guard-20260606

## Context

The sequence planner emits future `apply_operator_proposal` requests from ready
preflight results. Those results can become stale after queue changes.

## Options Considered

1. Trust ready preflight results without checking the queue.
2. Require current pending queue state before planning each apply request.
3. Re-run batch preflight inside the planner.

## Decision

Use option 2. A ready preflight result may become a plan step only when the
current queue still contains the same pending, non-expired, applyable proposal.

## Why This Won

It prevents stale or forged preflight evidence from becoming a supervised apply
request while keeping planning read-only and predictable.

## Rejected Alternatives

- Trusting preflight results was rejected because proposal state can change
  after preflight.
- Hidden preflight was rejected because planning should not query pre-read state
  or add surprise latency.
- A sequence executor was deferred until the planner has a stronger stale-state
  contract.

## Implementation Plan

1. Add a current-queue validation helper to the planner.
2. Block missing, non-pending, expired, no-longer-applyable, or changed
   proposals.
3. Include the current proposal snapshot in planned steps and blocked evidence.
4. Surface `current_queue_proposal_required` and `queue_freshness_checked` in
   payloads and Decision Queue visibility.
5. Add focused tests for valid, forged, and stale ready results.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_builds_read_only_plan_from_batch_preflight -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_rejects_forged_ready_result -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_blocks_stale_non_pending_result -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: old preflight payloads will no longer plan if the proposal has left the
queue. This is the intended fail-closed behavior.

Rollback: remove the current-queue guard and related payload fields. The base
sequence planner remains read-only, but with weaker stale-state protection.
