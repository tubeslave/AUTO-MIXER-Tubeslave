# ADR: Task Details Blocker List Blocker Guard

## Context

M040 blocked the Task Details apply entrypoint while offline integration bundle
blockers are visible, and M044 added a visible blocked status badge. The
`Apply Blockers` list still rendered only task-local blockers, so a task with
no local `applyBlockers` could show a blocked status without listing the global
integration-bundle reason in the detail panel.

## Options Considered

1. Leave the blocker list unchanged and rely on the status badge.
2. Add the global integration-bundle blocker to the Task Details blocker list
   while `integrationBundleApplyBlocked` is true.
3. Mutate task `applyBlockers` or backend task data before rendering.

## Decision

Derive a local `detailApplyBlockers` list. When the global blocker is visible,
prepend `offline_integration_bundle_blockers_visible` and de-duplicate it with
the task-local blockers; otherwise keep the existing task-local blocker list.

This does not change task data, hide rows, queue proposals, open Supervised
Apply, dispatch supervised apply, contact backend/WING, grant live authority or
close readiness.

## Why This Won

The detail panel should explain why the selected task is blocked at the same
place it already lists apply blockers. A local derived list keeps the UI
fail-closed and operator-readable without modifying the task model or backend
safety gates.

## Rejected Alternatives

Leaving the list unchanged would keep a confusing gap between the red `Blocked`
badge and an absent blocker list. Mutating task data was rejected because this
checkpoint is frontend presentation only and cannot claim backend, hardware or
operator proof.

## Implementation Plan

1. Add `detailApplyBlockers` inside `TaskDetailsPanel`.
2. Include `offline_integration_bundle_blockers_visible` while
   `integrationBundleApplyBlocked` is true.
3. Render the existing `Apply Blockers` section from `detailApplyBlockers`.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m046_task_details_blocker_list_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Details blocker-list presentation. If it
regresses, revert the derived blocker list and M046 evidence/test/docs updates.
The rollback does not alter backend gates, WING transport or accepted hardware
evidence.
