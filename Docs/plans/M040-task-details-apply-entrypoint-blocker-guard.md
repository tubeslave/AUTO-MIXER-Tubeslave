# Plan M040: Task Details apply entrypoint blocker guard

## Goal

Make the Task Details apply button stay blocked before opening Supervised
Apply when offline integration bundle blockers are visible.

## Scope

- `frontend/src/views/TaskQueueView.tsx`
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
| M040-CP01 | Pass integration bundle blocked state into `TaskDetailsPanel` and keep the detail apply button blocked | `TaskQueueView.tsx` and `TaskQueuePanels.tsx` source guard | planned |
| M040-CP02 | Add contract and rail coverage for the Task Details apply entrypoint | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m040_task_details_apply_entrypoint_is_guarded_by_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the `TaskDetailsPanel` blocked-state prop, M040 frontend contract test,
M040 rail, roadmap row, ADR, completion-gate paragraph and autopilot evidence
updates. Rollback is local and does not touch backend safety gates or hardware
evidence.
