# ADR: product-layer-read-surface-next-target-consistency-20260605

## Context

The no-WING read-surface summary now contains
`next_physical_readback_surface_id`. Product Layer consumes that summary for
cockpit proof evidence, so it must not silently replace a malformed source value
with a derived one.

## Options considered

1. Compare source target with the first physical gap and expose a consistency
   flag.
2. Trust the source target without checking.
3. Ignore the source target and keep deriving downstream.

## Decision

Use option 1. Product Layer exposes the source target, a display-safe fallback
target, and `next_physical_readback_target_consistent`.

## Why this won

It keeps cockpit evidence actionable while making artifact drift visible and
read-only.

## Rejected alternatives

Trusting the source hides malformed artifacts. Ignoring the source duplicates
release-artifact normalization downstream.

## Implementation plan

1. Add source target extraction and consistency comparison in Product Layer.
2. Add backend tests for matching and mismatched summary targets.
3. Carry the flag through frontend module profiles.
4. Render the gate in Settings and cover it in e2e.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q`
- `node frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `npm --prefix frontend run build`
- `npm --prefix frontend run test:e2e -- automixer-prototype.spec.ts`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: a mismatched artifact still shows a fallback target. The UI separates the
target label from the gate state, and no live action is added.

Rollback is removing the consistency fields, frontend gate row, tests, and this
ADR.
