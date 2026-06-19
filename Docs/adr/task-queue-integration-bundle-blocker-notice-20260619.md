# ADR: Task Queue Integration Bundle Blocker Notice

## Context

M029-M034 made Director surfaces show offline integration bundle blockers as
read-only operator evidence. The Task Queue is closer to accept/apply review
than the Director panel, so a blocked offline integration bundle also needs to
be visible there before proposal actions.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only Task Queue display only.

## Options Considered

1. Leave the Task Queue unchanged and rely on Director surfaces.
2. Show a blocker notice inside Task Queue while keeping proposal rows
   inspectable.
3. Disable or hide all Task Queue actions whenever the offline integration
   bundle is blocked.

## Decision

Show a read-only Task Queue notice:
`Promotion planning blocked. Task Queue evidence remains review-only` when
Director runtime blocked-reason messages and task rows are visible.

This does not queue proposals, hide review-only task rows, contact
backend/WING, send transport, grant readiness closure or grant live authority.

## Why This Won

The operator should see the blocker at the point where proposal actions are
reviewed. A local notice keeps the Task Queue truthful while preserving the
audit value of queued rows and existing inspect/reject controls.

## Rejected Alternatives

Leaving the Task Queue unchanged was rejected because it makes the operator
cross-read Director surfaces before reviewing queued tasks.

Hiding or disabling all rows was rejected because review-only evidence should
remain inspectable and because live eligibility must be governed by explicit
proposal, proof and hardware gates rather than a broad UI disappearance.

## Implementation Plan

- Pass `data.directorRuntime` into `TaskQueueWorkspace`.
- Render a blocker notice in `frontend/src/components/prototype/TaskQueuePanels.tsx`.
- Extend
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to assert
  the notice wiring in the active Task Queue source.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m035_task_queue_shows_integration_bundle_blocker_notice -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: the notice could be read as disabling every historical queue row. The
wording ties it to promotion planning and evidence review; existing row actions
still rely on their own apply gates.

Rollback: revert the M035 Task Queue panel/view/test/docs/state changes. The
M026-M034 readiness and Director surfaces remain available.
