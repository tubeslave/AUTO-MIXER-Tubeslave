# ADR: Task Queue Review Filter Blocker Guard

## Context

M041 made the Task Queue apply-capacity metric fail closed while offline
integration bundle blockers are visible. The `Review only` table filter still
used only per-task `reviewOnly` and `canApply` flags. That could leave some
rows outside the `Review only` filter even though the global blocker made all
apply entrypoints unavailable.

## Options Considered

1. Leave the filter unchanged and rely on the blocker notice plus action guards.
2. Include the integration bundle blocked state in the `Review only` predicate.
3. Add backend mutation changes in this checkpoint.

## Decision

Make the `Review only` predicate return true for every row while
`integrationBundleApplyBlocked` is true. Keep the `All`, `High`, `Pending` and
`Applied` filters unchanged.

This does not hide rows, queue proposals, open Supervised Apply, dispatch
supervised apply, contact backend/WING, grant live authority or close readiness.

## Why This Won

The table filters should match the same global blocked state already shown by
the notice, actions and metrics. Making `Review only` include every row while
the global blocker is active keeps operator scanning consistent without
changing backend safety gates or row data.

## Rejected Alternatives

Leaving the filter unchanged would preserve a subtle inconsistent operator
surface. Backend mutation changes were rejected because this checkpoint only
corrects local frontend filtering and cannot claim backend or hardware proof.

## Implementation Plan

1. Add `integrationBundleApplyBlocked` to the `Review only` filter predicate.
2. Keep the existing per-task review-only predicate for the unblocked case.
3. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m042_task_queue_review_filter_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Queue filtering. If it regresses, revert the
predicate change and M042 evidence/test/docs updates. The rollback does not
alter backend gates, WING transport or accepted hardware evidence.
