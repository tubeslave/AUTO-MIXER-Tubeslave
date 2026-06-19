# M037 Supervised Apply Integration Bundle Blocker Enforcement

## Goal

Make the Supervised Apply UI enforce the blocked offline integration bundle
state, so a visible promotion-planning blocker cannot still leave apply confirm
controls available in the frontend.

## Scope

- Frontend Task Queue and Supervised Apply modal only.
- Offline/control-plane display and local UI gating only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M037-CP01 Supervised Apply Blocked-State Enforcement

Update `frontend/src/components/prototype/TaskQueuePanels.tsx` so visible
Director runtime integration-bundle blockers make task-row apply actions
unavailable and make `SupervisedApplyModal` disable apply confirm through the
same local `canRequestBackendGate` path.

### M037-CP02 Contract Test and Evidence Sync

Extend the frontend operator adapter contract test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M037 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Task rows do not expose the Apply action while integration-bundle blockers
  are visible.
- Supervised Apply modal marks the gate as blocked by integration bundle.
- Supervised Apply confirm is disabled unless the existing backend gate and
  integration-bundle blocker checks both pass.
- No backend contact, queue mutation or live transport is introduced.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m037_supervised_apply_blocker_enforcement_blocks_apply_confirm -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M037 Supervised Apply enforcement/test and accompanying docs/state
evidence. The M026-M036 readiness, Director, Agent Suggestions, Task Queue and
Supervised Apply notice surfaces remain intact.
