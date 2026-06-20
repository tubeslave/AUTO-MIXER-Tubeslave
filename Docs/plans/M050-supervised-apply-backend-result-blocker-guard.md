# Plan M050: Supervised Apply backend result blocker guard

## Goal

Make the Supervised Apply modal hide matched backend result rows and the
`Backend Result` panel while promotion planning is blocked.

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
- No queue mutation.
- No supervised apply opening or dispatch.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

| Checkpoint | Objective | Evidence | Status |
| --- | --- | --- | --- |
| M050-CP01 | Hide matched backend result presentation while integration bundle apply is blocked | `TaskQueuePanels.tsx` local `visibleBackendResultMatches` guard | planned |
| M050-CP02 | Add contract and rail coverage for the hidden backend result presentation | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m050_supervised_apply_backend_result_is_hidden_by_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the `visibleBackendResultMatches` guard, M050 frontend contract test,
M050 rail, roadmap row, ADR, completion-gate paragraph and autopilot evidence
updates. Rollback is local and does not touch backend safety gates or hardware
evidence.
