# Plan M057: Safety Monitor goal readiness blocker guard

## Goal

Make the Safety Monitor `Goal readiness` row show an explicit blocked state
while promotion planning is blocked by offline integration bundle evidence.

## Scope

- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `tests/test_wing_mvp_rails.py`
- roadmap, capability, completion-gate, ADR and autopilot evidence updates

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No readiness input mutation.
- No safety monitor input mutation.
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
| M057-CP01 | Render blocked Safety Monitor goal readiness while integration bundle apply is blocked | `TaskQueuePanels.tsx` local `goalReadinessStatus` guard | planned |
| M057-CP02 | Add contract and rail coverage for blocked Safety Monitor goal-readiness presentation | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m057_safety_monitor_goal_readiness_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the Safety Monitor goal-readiness presentation guard, M057 frontend
contract test, M057 rail, roadmap row, ADR, completion-gate paragraph and
autopilot evidence updates. Rollback is local and does not touch backend safety
gates or hardware evidence.
