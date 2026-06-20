# Plan M058: Safety Monitor blocked surfaces blocker guard

## Goal

Make the Safety Monitor `Blocked` footer show the visible offline integration
bundle blocker while promotion planning is blocked.

## Scope

- `frontend/src/components/prototype/TaskQueuePanels.tsx`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `tests/test_wing_mvp_rails.py`
- roadmap, capability, completion-gate, ADR and autopilot evidence updates

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No safety monitor input mutation.
- No readiness input mutation.
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
| M058-CP01 | Render the integration bundle blocker in Safety Monitor blocked surfaces while promotion planning is blocked | `TaskQueuePanels.tsx` local `blockedSurfaceText` guard | planned |
| M058-CP02 | Add contract and rail coverage for blocked Safety Monitor footer presentation | frontend contract test plus `test_wing_mvp_rails.py` | planned |

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m058_safety_monitor_blocked_surfaces_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Rollback

Revert the Safety Monitor blocked-surface presentation guard, M058 frontend
contract test, M058 rail, roadmap row, ADR, completion-gate paragraph and
autopilot evidence updates. Rollback is local and does not touch backend safety
gates or hardware evidence.
