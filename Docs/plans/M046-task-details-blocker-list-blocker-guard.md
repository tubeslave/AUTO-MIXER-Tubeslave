# Plan M046: Task Details blocker list blocker guard

## Goal

Make the Task Details `Apply Blockers` list include the offline integration
bundle blocker when promotion planning is blocked, even if the selected task has
no local apply blockers.

## Scope

- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `tests/test_wing_mvp_rails.py`
- roadmap, capability, completion-gate, ADR and autopilot evidence updates

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
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
| M046-CP01 | Make the Task Details blocker list include the integration bundle blocker while apply is globally blocked | `TaskQueuePanels.tsx` `detailApplyBlockers` guard | planned |
| M046-CP02 | Add contract and rail coverage for the blocked Task Details blocker list | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m046_task_details_blocker_list_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the Task Details blocker-list guard, M046 frontend contract test, M046
rail, roadmap row, ADR, completion-gate paragraph and autopilot evidence
updates. Rollback is local and does not touch backend safety gates or hardware
evidence.
