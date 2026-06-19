# Plan M043: Task Queue row badge blocker guard

## Goal

Make Task Queue row badges show the global blocked state when offline
integration bundle blockers make every apply path review-only.

## Scope

- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `tests/test_wing_mvp_rails.py`
- roadmap, capability, completion-gate, ADR and autopilot evidence updates

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No queue mutation.
- No supervised apply dispatch.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

| Checkpoint | Objective | Evidence | Status |
| --- | --- | --- | --- |
| M043-CP01 | Make row badges fail closed while integration bundle apply is blocked | `TaskQueuePanels.tsx` row badge guard | planned |
| M043-CP02 | Add contract and rail coverage for blocked Task Queue row badges | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m043_task_queue_row_badge_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the Task Queue row-badge guard, M043 frontend contract test, M043 rail,
roadmap row, ADR, completion-gate paragraph and autopilot evidence updates.
Rollback is local and does not touch backend safety gates or hardware evidence.
