# M038 Supervised Apply Confirm Callback Blocker Guard

## Goal

Make the Supervised Apply confirm callback fail closed when the offline
integration bundle is blocked, so a disabled UI control is not the only
protection against frontend dispatch.

## Scope

- Frontend Task Queue and Supervised Apply modal callback only.
- Offline/control-plane local UI dispatch guard only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M038-CP01 Guarded Confirm Callback

Update `frontend/src/components/prototype/TaskQueuePanels.tsx` so
`onConfirm` calls a guarded local function. The function returns before
dispatching `actions.confirmSupervisedApply` while integration-bundle blockers
are visible.

### M038-CP02 Contract Test and Evidence Sync

Extend the frontend operator adapter contract test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M038 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Supervised Apply confirm callback calls a guarded function, not the action
  directly.
- The guarded function returns before dispatch while integration-bundle
  blockers are visible.
- The guarded function dispatches the existing action only when the blocker is
  absent.
- No backend contact, queue mutation or live transport is introduced.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m038_supervised_apply_confirm_callback_is_guarded_by_blocker -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M038 guarded-callback/test and accompanying docs/state evidence. The
M026-M037 readiness, Director, Agent Suggestions, Task Queue, Supervised Apply
notice and Supervised Apply enforcement surfaces remain intact.
