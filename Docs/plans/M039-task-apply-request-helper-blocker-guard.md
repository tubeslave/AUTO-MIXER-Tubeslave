# Plan M039: Task apply request helper blocker guard

## Goal

Make the Task Queue apply request helper fail closed before opening the
Supervised Apply modal when offline integration bundle blockers are visible.

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
| M039-CP01 | Guard `requestTaskApply` before it opens Supervised Apply while integration bundle blockers are visible | `TaskQueuePanels.tsx` source guard | planned |
| M039-CP02 | Add contract and rail coverage for helper-level blocked apply requests | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m039_task_apply_request_helper_is_guarded_by_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the helper guard, M039 frontend contract test, M039 rail, roadmap row,
ADR, completion-gate paragraph and autopilot evidence updates. Rollback is
local and does not touch backend safety gates or hardware evidence.
