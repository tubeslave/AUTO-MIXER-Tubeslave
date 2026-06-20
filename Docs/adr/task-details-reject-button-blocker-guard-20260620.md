# ADR: Task Details Reject Button Blocker Guard

## Context

M053 hid row-level `Reject` while offline integration bundle blockers are
visible. The Task Details panel still exposed a `Reject` button that could
mutate task status even though the surrounding blocked state says promotion
planning evidence remains review-only.

## Options Considered

1. Leave Task Details `Reject` active because it is local and not a live console
   action.
2. Disable Task Details `Reject` only when the global integration bundle blocker
   is visible, while preserving normal reject behavior when the blocker is
   absent.
3. Remove all reject affordances from Task Queue permanently.

## Decision

Add local Task Details reject button presentation state derived from
`integrationBundleApplyBlocked`. When the global blocker is visible, the panel
shows a disabled `Reject Blocked` button with danger styling. When the blocker
is absent, the existing `Reject` behavior remains unchanged.

This does not mutate task data, backend result input, backend status input,
queue proposals, open Supervised Apply, dispatch supervised apply, contact
backend/WING, grant live authority or close readiness.

## Why This Won

Task Details is the focused review surface for a proposal. In a globally blocked
promotion-planning state, its bottom action row should not offer a task-status
mutation while every surrounding gate explains that the evidence is read-only.
Disabling only the global-blocker case keeps normal local review behavior intact.

## Rejected Alternatives

Leaving `Reject` active kept a local mutation on a blocked evidence surface.
Removing all reject affordances permanently was rejected because ordinary local
review-only tasks still need normal operator triage when global promotion
planning is not blocked.

## Implementation Plan

1. Add Task Details reject button label and class derivation from
   `integrationBundleApplyBlocked`.
2. Render disabled `Reject Blocked` for the global blocker.
3. Preserve existing Task Details reject behavior when the blocker is absent.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m054_task_details_reject_button_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Details button presentation. If it regresses,
revert the local reject button presentation guard and M054 evidence/test/docs
updates. The rollback does not alter backend gates, WING transport or accepted
hardware evidence.
