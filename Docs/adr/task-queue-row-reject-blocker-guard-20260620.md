# ADR: Task Queue Row Reject Blocker Guard

## Context

M053 follows the blocked Task Queue presentation work from M035 through M052.
The table row actions already removed `Apply` while offline integration bundle
blockers were visible, but still exposed `Reject`. That action is local and not
a console write, yet it can mutate task status on a surface that says evidence
remains review-only.

## Options Considered

1. Leave row `Reject` visible while blocked because it is not a live console
   action.
2. Hide row `Reject` only when the global integration bundle blocker is visible,
   while preserving existing `Reject` for local review-only tasks.
3. Disable or remove every reject button in Task Queue and Task Details.

## Decision

Add a local `taskQueueRowActions` helper. When
`integrationBundleApplyBlocked` is true, table rows expose only `Detail`. When
the global blocker is absent, local review-only rows still expose `Detail` and
`Reject`, and applyable rows keep `Detail`, `Apply` and `Reject`.

This does not mutate task data, backend result input, backend status input,
queue proposals, open Supervised Apply, dispatch supervised apply, contact
backend/WING, grant live authority or close readiness.

## Why This Won

The row action cluster is the fastest operator path for changing queue state.
When promotion planning is globally blocked, row-level actions should be
inspection-only so evidence rows cannot be changed or hidden from the blocked
review path by an unrelated status mutation.

## Rejected Alternatives

Leaving `Reject` visible kept a local mutation on a read-only evidence surface.
Removing all reject affordances everywhere was rejected because this checkpoint
only covers the row action cluster and should not change Task Details review
behavior without a separate narrow checkpoint.

## Implementation Plan

1. Add `taskQueueRowActions` in `TaskQueuePanels.tsx`.
2. Return only `Detail` while `integrationBundleApplyBlocked` is true.
3. Preserve existing row actions when the global blocker is absent.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m053_task_queue_row_reject_action_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Queue row action presentation. If it regresses,
revert the local row action guard and M053 evidence/test/docs updates. The
rollback does not alter backend gates, WING transport or accepted hardware
evidence.
