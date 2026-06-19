# ADR: Task Queue Row Badge Blocker Guard

## Context

M042 made the Task Queue `Review only` filter include every row while offline
integration bundle blockers are visible. The row-level task badge still used
only each task's `reviewOnly` flag, so rows could remain visually unmarked even
though the same global blocker made their apply action unavailable.

## Options Considered

1. Leave the row badge unchanged and rely on the blocker notice, metrics and actions.
2. Show a blocked row badge whenever `integrationBundleApplyBlocked` is true.
3. Add backend row-data mutations in this checkpoint.

## Decision

Show a red `Blocked` badge on every Task Queue row while
`integrationBundleApplyBlocked` is true. Keep the existing per-task `Review`
badge for tasks that are review-only when the global blocker is absent.

This does not change row data, hide rows, queue proposals, open Supervised
Apply, dispatch supervised apply, contact backend/WING, grant live authority or
close readiness.

## Why This Won

The row badge is the closest visual cue to each task title. Matching it to the
global blocked state keeps operator scanning consistent with the notice,
actions, metrics and filter without changing backend safety gates or task data.

## Rejected Alternatives

Leaving the row badge unchanged would preserve an inconsistent operator-facing
surface. Backend row-data changes were rejected because this checkpoint only
corrects local frontend presentation and cannot claim backend or hardware
proof.

## Implementation Plan

1. Include `integrationBundleApplyBlocked` in the Task Queue row badge guard.
2. Render `Blocked` when the global blocker is visible and `Review` for the
   existing per-task review-only case.
3. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m043_task_queue_row_badge_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Queue presentation. If it regresses, revert
the row-badge guard and M043 evidence/test/docs updates. The rollback does not
alter backend gates, WING transport or accepted hardware evidence.
