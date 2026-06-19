# M030 Director Event Integration Bundle Blocker Summary

## Goal

Render M029 `DirectorRuntime.blockedReasons` integration bundle tags as an
operator-readable Director event so the cockpit event stream explains the
offline integration blocker without exposing raw tag syntax.

## Scope

- Frontend Director panel model only.
- Offline/control-plane display only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M030-CP01 Director Event Blocker Summary

Update `frontend/src/components/dashboard/directorPanelModel.ts` so
`directorEventRows(...)` detects `integration_bundle_*` blocked-reason tags
and emits one red `Integration bundle` event with:

- integration bundle status;
- blocker count;
- missing integration checks;
- source authority, live authority and readiness-closure denial.

Raw integration bundle tags must not be repeated in the generic `Blockers`
event when no other blockers exist.

### M030-CP02 Contract Test and Evidence Sync

Add a frontend contract test for the Director event summary, update roadmap,
completion gate, capability matrix, current state and autopilot evidence so
M030 is durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Backend Director events include a readable `Integration bundle` event when
  integration bundle blocker tags are present.
- Raw `integration_bundle_*` tags are not exposed as the only generic blocker
  event.
- Non-integration blockers still render through the generic blocker event.
- The Director event surface remains read-only.
- Event-visible integration bundle status cannot grant live authority,
  readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m030_director_events_render_integration_bundle_blocker_summary -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M030 Director event model/test and accompanying docs/state evidence.
The M026-M029 readiness surfaces remain intact.
