# ADR: operator-proposal-batch-preflight-20260606

## Context

Operators can inspect queue preflight candidates and dry-run a single proposal
apply. A batch endpoint is needed to validate the current candidate set before
any supervised apply sequence.

## Options Considered

1. Run batch preflight automatically from `build_decision_queue`.
2. Add a Product Layer batch preflight handler.
3. Automatically apply all proposals that pass preflight.

## Decision

Use option 2. Add `preflight_operator_proposal_apply_batch` as an explicit
operator-requested dry-run handler.

## Why This Won

It makes the pending apply set explainable without adding side effects to queue
snapshots and without widening live execution.

## Rejected Alternatives

- Automatic snapshot preflight was rejected because it could query pre-read
  state from a read-only view call.
- Automatic apply was rejected because preflight readiness is not operator
  approval and not proof of a successful write/readback/rollback sequence.
- A separate service extraction was deferred until the Product Layer contract is
  stable.

## Implementation Plan

1. Keep cumulative-delta validation non-mutating in preflight mode.
2. Add the batch handler and dispatch registration.
3. Select pending apply candidates or explicit proposal IDs.
4. Return ready/blocked/skipped counts and per-proposal payloads.
5. Test no live writes, no queue mutation, no idempotency reservation, and no
   delta-history mutation.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_batch_preflight_operator_proposal_apply_reports_ready_and_blocked_without_live_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: batch preflight may be mistaken for permission to apply all ready
proposals. Mitigation: payload uses `batch_preflight_only=true`,
`apply_attempted=false`, and `live_apply_entrypoint=apply_operator_proposal`.

Rollback: remove the batch handler and dispatch registration. Single-proposal
preflight and apply behavior remain unchanged.
