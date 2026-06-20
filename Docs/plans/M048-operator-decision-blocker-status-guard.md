# Plan M048: Operator Decision blocker status guard

## Goal

Make the Task Queue `Operator Decision` panel show blocked integration-bundle
status while promotion planning is blocked.

## Scope

- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `tests/test_wing_mvp_rails.py`
- roadmap, capability, completion-gate, ADR and autopilot evidence updates

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No backend status input mutation.
- No task data mutation.
- No queue mutation.
- No supervised apply opening or dispatch.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

| Checkpoint | Objective | Evidence | Status |
| --- | --- | --- | --- |
| M048-CP01 | Derive blocked Operator Decision write-path and backend statuses while integration bundle apply is blocked | `TaskQueuePanels.tsx` local `writePathStatus` and `backendStatus` guards | planned |
| M048-CP02 | Add contract and rail coverage for the blocked Operator Decision status surface | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m048_operator_decision_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the derived Operator Decision blocker statuses, M048 frontend contract
test, M048 rail, roadmap row, ADR, completion-gate paragraph and autopilot
evidence updates. Rollback is local and does not touch backend safety gates or
hardware evidence.
