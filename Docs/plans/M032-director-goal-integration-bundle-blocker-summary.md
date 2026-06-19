# M032 Director Goal Integration Bundle Blocker Summary

## Goal

Make the Director runtime `currentGoal` surface the offline integration bundle
blocker when `main_pipeline_preview.summary` reports that the bundle is not
ready, so the operator sees the active promotion-planning blocker before the
generic supervised-gate fallback.

## Scope

- Frontend Director runtime adapter only.
- Offline/control-plane display only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M032-CP01 Director Goal Blocker Priority

Update `frontend/src/services/operatorAdapters.ts` so emergency stop still
wins, then an integration bundle blocker sets `DirectorRuntime.currentGoal`,
then the supervised-gate and generic fallback goals apply.

### M032-CP02 Contract Test and Evidence Sync

Extend the frontend operator adapter contract test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M032 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Blocked offline integration bundle preview data changes `currentGoal` to a
  blocker-focused read-only message.
- Emergency stop remains the highest-priority goal.
- The goal text says proposals remain review-only and no live authority,
  readiness closure or console apply is granted.
- The Director goal remains read-only.
- Goal-visible integration bundle status cannot grant live authority,
  readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m032_director_goal_surfaces_integration_bundle_blocker_summary -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M032 Director runtime adapter/test and accompanying docs/state
evidence. The M026-M031 readiness, event and banner surfaces remain intact.
