# ADR: decision-queue-batch-preflight-visibility-20260606

## Context

`preflight_operator_proposal_apply_batch` can dry-run the current pending apply
candidate set on explicit operator request. The decision queue snapshot does
not yet expose that batch preflight path as read-only UI evidence.

## Options considered

1. Run batch preflight automatically inside `build_decision_queue`.
2. Add snapshot-only batch-preflight visibility to the decision queue.
3. Wait for frontend to infer batch behavior from pending proposal rows.

## Decision

Use option 2. Add `decision_queue.summary.batch_preflight` as a read-only
description of the explicit batch preflight handler and default candidate set.

## Why this won

It gives operators and frontend code a backend-owned contract without running
preflight, mutating queue state, reserving idempotency keys, or touching mixer
clients.

## Rejected alternatives

- Automatic batch preflight was rejected because queue snapshots must be cheap
  and side-effect free.
- Frontend inference was rejected because the backend should publish safety
  flags and handler metadata explicitly.
- Treating batch candidates as ready-to-apply was rejected because
  `apply_operator_proposal` remains the only supervised live apply entrypoint.

## Implementation plan

1. Add a Product Layer helper that splits pending proposals into default batch
   candidates and analysis-only skipped rows.
2. Attach the helper result to `decision_queue.summary`.
3. Add top-level count aliases for frontend adapters.
4. Extend focused tests for snapshot-only no-live-write behavior.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_decision_queue_summarizes_apply_preflight_candidates -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: the UI may overstate batch-preflight candidates as apply-ready.
Mitigation: publish `runs_preflight=false`, `apply_attempted=false`,
`queue_mutation=false`, and `live_apply_entrypoint=apply_operator_proposal`.

Rollback: remove `batch_preflight` summary fields and the focused test
assertions. The batch handler and existing proposal apply flow remain intact.
