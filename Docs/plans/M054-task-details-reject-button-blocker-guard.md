# Plan M054: Task Details reject button blocker guard

## Goal

Make the Task Details reject button show an explicit blocked state while
promotion planning is blocked by offline integration bundle evidence.

## Scope

- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `tests/test_wing_mvp_rails.py`
- roadmap, capability, completion-gate, ADR and autopilot evidence updates

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No backend result input mutation.
- No backend status input mutation.
- No task data mutation.
- No row hiding.
- No queue mutation.
- No supervised apply opening or dispatch.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

| Checkpoint | Objective | Evidence | Status |
| --- | --- | --- | --- |
| M054-CP01 | Render a disabled blocked Task Details reject button while integration bundle apply is blocked | `TaskQueuePanels.tsx` local `detailRejectButtonLabel` and `detailRejectButtonClassName` guard | planned |
| M054-CP02 | Add contract and rail coverage for the blocked Task Details reject button presentation | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m054_task_details_reject_button_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the Task Details reject button presentation guard, M054 frontend
contract test, M054 rail, roadmap row, ADR, completion-gate paragraph and
autopilot evidence updates. Rollback is local and does not touch backend safety
gates or hardware evidence.
