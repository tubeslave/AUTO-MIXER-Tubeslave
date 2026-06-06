# Brief: operator-proposal-apply-sequence-queue-guard-20260606

## Context

`plan_operator_proposal_apply_sequence` can turn ready preflight results into
ordered supervised apply requests. A stale result could otherwise survive after
the proposal was accepted, dismissed, applied, expired, or changed.

## Objective

Require the sequence planner to verify every ready preflight result against the
current `OperatorProposalQueue` before emitting any apply request.

## Safety Constraints

- Do not run preflight from the planner.
- Do not call `_apply_manual_console_write`.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal status, or delta history.
- Reject ready results whose proposal is missing, non-pending, expired, no
  longer applyable, or changed since preflight.

## Expected Evidence

- Valid ready results for current pending proposals still become plan steps.
- Stale ready results for non-pending proposals are blocked.
- Forged ready results without a queue proposal are blocked.
- The planner reports that current queue freshness was checked.
