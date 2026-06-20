# ADR: Task Details Badge Blocker Guard

## Context

M043 made Task Queue rows show a blocked badge while offline integration bundle
blockers are visible. The selected Task Details panel still displayed its
review badge only from the per-task `reviewOnly` flag, so the detail view could
look less blocked than the table row and apply button.

## Options Considered

1. Leave the Task Details badge unchanged and rely on the row badge plus apply button.
2. Show a blocked Task Details badge whenever `integrationBundleApplyBlocked` is true.
3. Mutate task status or backend task data in this checkpoint.

## Decision

Show a red `Blocked` badge in Task Details while `integrationBundleApplyBlocked`
is true. Keep the existing per-task `Review only` badge when the global blocker
is absent.

This does not change task data, hide rows, queue proposals, open Supervised
Apply, dispatch supervised apply, contact backend/WING, grant live authority or
close readiness.

## Why This Won

The detail panel is where an operator inspects the selected proposal before any
apply path. Matching its status badges to the global blocked state keeps the
selected-task view consistent with the row badge, filter, metrics and blocked
apply entrypoint without changing backend safety gates.

## Rejected Alternatives

Leaving the detail badge unchanged would preserve a subtle inconsistent
operator surface. Mutating status or backend task data was rejected because this
checkpoint only corrects local frontend presentation and cannot claim backend
or hardware proof.

## Implementation Plan

1. Include `integrationBundleApplyBlocked` in the Task Details review badge guard.
2. Render `Blocked` when the global blocker is visible and `Review only` for
   the existing per-task review-only case.
3. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m044_task_details_badge_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Details presentation. If it regresses, revert
the detail-badge guard and M044 evidence/test/docs updates. The rollback does
not alter backend gates, WING transport or accepted hardware evidence.
