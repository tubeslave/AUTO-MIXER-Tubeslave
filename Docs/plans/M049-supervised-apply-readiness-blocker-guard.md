# Plan M049: Supervised Apply readiness blocker guard

## Goal

Make the Supervised Apply modal `Final readiness` row show blocked
integration-bundle status while promotion planning is blocked.

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
| M049-CP01 | Derive blocked Supervised Apply final-readiness status while integration bundle apply is blocked | `TaskQueuePanels.tsx` local `finalReadinessStatus` guard | planned |
| M049-CP02 | Add contract and rail coverage for the blocked Supervised Apply final-readiness row | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m049_supervised_apply_final_readiness_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the derived Supervised Apply final-readiness status, M049 frontend
contract test, M049 rail, roadmap row, ADR, completion-gate paragraph and
autopilot evidence updates. Rollback is local and does not touch backend safety
gates or hardware evidence.
