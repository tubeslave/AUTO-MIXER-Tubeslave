# M031 Director Guard Integration Bundle Blocker Banner

## Goal

Render M029 `DirectorRuntime.blockedReasons` integration bundle tags as
operator-readable Runtime Guard banner messages so the Director panel does not
surface raw `integration_bundle_*` tag syntax to the operator.

## Scope

- Frontend Director panel model and Runtime Guard banner only.
- Offline/control-plane display only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M031-CP01 Runtime Guard Blocker Message Formatter

Export a shared Director blocked-reason formatter from
`frontend/src/components/dashboard/directorPanelModel.ts` and use it in
`frontend/src/views/DirectorPanel.tsx` so the Runtime Guard blocker banner
shows readable integration bundle status and denial details.

### M031-CP02 Contract Test and Evidence Sync

Add a frontend contract test for the banner formatter/use-site, update roadmap,
completion gate, capability matrix, current state and autopilot evidence so
M031 is durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Runtime Guard blocker banner uses readable blocked-reason messages.
- Integration bundle blocker tags render as one readable message.
- Non-integration blockers remain visible.
- Raw `integration_bundle_*` tags are not surfaced by the banner.
- The Director guard banner remains read-only.
- Banner-visible integration bundle status cannot grant live authority,
  readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m031_director_guard_banner_renders_integration_bundle_blocker_summary -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M031 Director panel/model/test and accompanying docs/state evidence.
The M026-M030 readiness and event surfaces remain intact.
