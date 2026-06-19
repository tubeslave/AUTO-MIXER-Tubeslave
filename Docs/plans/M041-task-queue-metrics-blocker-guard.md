# Plan M041: Task Queue metrics blocker guard

## Goal

Make Task Queue metrics show blocked apply capacity when offline integration
bundle blockers are visible.

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
| M041-CP01 | Make Task Queue metrics fail closed while integration bundle apply is blocked | `TaskQueuePanels.tsx` metric guard | planned |
| M041-CP02 | Add contract and rail coverage for blocked Task Queue metrics | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m041_task_queue_metrics_reflect_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the Task Queue metric guard, M041 frontend contract test, M041 rail,
roadmap row, ADR, completion-gate paragraph and autopilot evidence updates.
Rollback is local and does not touch backend safety gates or hardware evidence.
