# ADR: Task Details Apply Button Blocker Guard

## Context

M040 kept the Task Details apply entrypoint from opening Supervised Apply while
offline integration bundle blockers are visible. The panel still did not show an
apply-specific blocked button state in that global blocker condition; it fell
back to the local inspection path. That was safe but less explicit for the
operator because the surrounding Task Queue surfaces already say promotion
planning is blocked.

## Options Considered

1. Leave the Task Details button as `Inspect` for every blocked task.
2. Show a disabled `Apply Blocked` button only when the global integration
   bundle blocker is visible, while preserving `Inspect` for ordinary local
   review-only tasks.
3. Call the existing apply request helper and rely on its blocked focus action.

## Decision

Add local Task Details button presentation state derived from
`integrationBundleApplyBlocked`. When the global blocker is visible, the panel
shows a disabled `Apply Blocked` button with danger styling. When the blocker is
absent, the existing active apply and local `Inspect` behavior remains
unchanged.

This does not mutate task data, backend result input, backend status input,
queue proposals, open Supervised Apply, dispatch supervised apply, contact
backend/WING, grant live authority or close readiness.

## Why This Won

The button is the final visible affordance in Task Details. In a globally
blocked state, it should not look like a normal apply affordance or hide the
reason behind a generic inspection path. A disabled `Apply Blocked` state keeps
the existing fail-closed behavior and makes the operator-facing state match the
rest of the blocked Task Queue UI.

## Rejected Alternatives

Leaving the button as `Inspect` was safe but less explicit. Calling the apply
request helper from a blocked button was rejected because the global blocker is
already known and the UI should not perform a blocked action to explain itself.

## Implementation Plan

1. Add Task Details apply button label and class derivation from
   `integrationBundleApplyBlocked`.
2. Render disabled `Apply Blocked` for the global blocker.
3. Preserve existing `Inspect` behavior for local task blockers.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m052_task_details_apply_button_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Details button presentation. If it regresses,
revert the local button presentation guard and M052 evidence/test/docs updates.
The rollback does not alter backend gates, WING transport or accepted hardware
evidence.
