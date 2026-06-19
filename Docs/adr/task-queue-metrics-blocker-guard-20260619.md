# ADR: Task Queue Metrics Blocker Guard

## Context

M037-M040 blocked the visible Task Queue apply actions, modal confirm callback,
shared apply request helper and detail-panel apply entrypoint when offline
integration bundle blockers are visible. The Task Queue metric strip still
reported applyable task capacity from per-task `canApply` alone, which could
show a green apply path while the global blocked state made all apply entrypoints
unavailable.

## Options Considered

1. Leave the metric unchanged and rely on action-level guards.
2. Pass the integration bundle blocked state into `taskQueueMetrics` and make
   apply capacity fail closed in the visible metric.
3. Add backend mutation changes in this checkpoint.

## Decision

Pass the integration bundle blocked state into `taskQueueMetrics`. When it is
true, report zero apply capacity and render the apply-capacity metric as
`Apply Blocked` with integration bundle context.

This does not queue proposals, open Supervised Apply, dispatch supervised
apply, contact backend/WING, grant live authority or close readiness.

## Why This Won

The metric strip is operator-facing state. It should agree with the blocked
apply controls so operators do not see a green capacity signal for actions the
system will refuse. Keeping the change in the frontend metric layer also avoids
expanding backend authority or changing safety gates.

## Rejected Alternatives

Leaving the metric unchanged would preserve an inconsistent signal. Backend
mutation changes were rejected because this checkpoint only corrects local
frontend presentation and cannot claim backend or hardware proof.

## Implementation Plan

1. Add an `integrationBundleApplyBlocked` argument to `taskQueueMetrics`.
2. Report zero apply capacity and blocked metric labeling while the argument is
   true.
3. Pass the Task Queue workspace blocked state into the metric helper.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m041_task_queue_metrics_reflect_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Queue metric presentation. If it regresses,
revert the metric argument, blocked metric formatting and M041 evidence/test/docs
changes. The rollback does not alter backend gates, WING transport or accepted
hardware evidence.
