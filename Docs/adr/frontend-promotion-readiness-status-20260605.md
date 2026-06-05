# ADR: frontend-promotion-readiness-status-20260605

## Context

`autonomy_promotion_readiness` is already part of the Product Layer processing
snapshot. It includes target-level blockers and evidence requirement IDs, but the
frontend Settings module status table does not render those rows.

## Options considered

1. Append promotion readiness rows to the existing module status adapter.
2. Build a new readiness panel now.
3. Keep readiness backend-only.

## Decision

Use option 1. The existing Settings table will show promotion readiness rows as
read-only status records.

## Why this won

It makes blocked autonomous targets visible to the operator with a small adapter
change and no new apply surface.

## Rejected alternatives

A new panel is more UI work than this evidence slice needs. Keeping readiness
backend-only leaves important blockers invisible.

## Implementation plan

1. Read `autonomy_promotion_readiness.rows` in `adaptModuleStatusRecords`.
2. Map rows to `promotion_readiness:*` module status records.
3. Keep `canWriteLive=false` for every readiness row.
4. Include blockers and evidence requirement IDs in display/search text.
5. Update frontend adapter contract tests.

## Test plan

- `npm --prefix frontend test -- --runInBand operatorAdaptersContract.test.mjs`
- `npm --prefix frontend test`
- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py -q`

## Risks and rollback

Risk: readiness status could be mistaken for live write authority. The adapter
sets `canWriteLive=false` and creates no action controls.

Rollback is removing the readiness row mapping, test fixture additions, and this
ADR.
