# ADR: Task Queue Review-Only Metric Blocker Guard

## Context

M041 made the Task Queue apply-capacity metric fail closed while offline
integration bundle blockers are visible. M042 made the `Review only` filter
include all rows under the same global blocker. The `Review Only` metric still
counted only per-task review-only rows, so the metric could under-report the
effective review-only queue while the global blocker made every row review-only.

## Options Considered

1. Leave the metric unchanged and rely on the `Apply Blocked` metric.
2. Count every row in the `Review Only` metric while `integrationBundleApplyBlocked` is true.
3. Mutate row status or backend task data in this checkpoint.

## Decision

Add a `reviewOnlyCount` derived from `integrationBundleApplyBlocked`. When the
global blocker is visible, `Review Only` uses `taskRows.length`; otherwise it
keeps the existing per-task `reviewOnly` or `canApply === false` count.

This does not change row data, hide rows, queue proposals, open Supervised
Apply, dispatch supervised apply, contact backend/WING, grant live authority or
close readiness.

## Why This Won

The metric row should match the same fail-closed state already shown by the
notice, apply-capacity metric, filter, row badges and detail badges. Counting
all rows while the global blocker is visible keeps operator scanning consistent
without changing backend safety gates or task data.

## Rejected Alternatives

Leaving the metric unchanged would keep a stale low count beside an otherwise
blocked Task Queue. Mutating row status or backend data was rejected because
this checkpoint only corrects local frontend presentation and cannot claim
backend or hardware proof.

## Implementation Plan

1. Add a `reviewOnlyCount` inside `taskQueueMetrics`.
2. Use `taskRows.length` when `integrationBundleApplyBlocked` is true.
3. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m045_task_queue_review_only_metric_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Queue metric presentation. If it regresses,
revert the metric count guard and M045 evidence/test/docs updates. The rollback
does not alter backend gates, WING transport or accepted hardware evidence.
