# ADR: autonomy-gap-summary-20260605

## Context

The autonomy readiness snapshot exposes target-level rows, but the project also
needs a compact way to see what still blocks broad autonomous console control.
Without an aggregate view, agents must parse every row to understand whether the
next gap is safety proof, read coverage, write-surface quarantine, or mode
policy.

## Options considered

1. Add `promotion_gap_summary` to `autonomy_promotion_readiness.summary`.
2. Create a separate remediation-plan snapshot.
3. Compute the summary only in the frontend.

## Decision

Use option 1. The summary belongs beside the readiness rows because it is a
pure aggregation of those rows.

Expose the same summary through the existing frontend Settings module-status
adapter as read-only rows.

## Why this won

It improves explainability and roadmap clarity without adding routes,
dependencies, or live-write authority.

## Rejected alternatives

A separate remediation-plan snapshot may be useful later, but it would duplicate
row parsing today. Frontend-only aggregation would make backend tests and future
agents depend on UI logic for canonical readiness evidence.

## Implementation plan

1. Categorize readiness blockers into stable gap buckets.
2. Count gaps and blocked target IDs per bucket.
3. Group evidence requirement IDs and next required evidence by action family.
4. Add the result under `summary.promotion_gap_summary`.
5. Extend readiness tests.
6. Map nonzero gap buckets to read-only frontend module status rows.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py -q`
- `npm --prefix frontend test -- --runInBand operatorAdaptersContract.test.mjs`

## Risks and rollback

Risk: aggregate counts could be mistaken for readiness approval. The snapshot
continues to set `live_write=false`, `live_widening_allowed=false`, and
`full_autonomous_ready=false`.

Rollback is removing the helper functions, summary field, tests, and this ADR.
