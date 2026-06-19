# M029 Director Runtime Integration Bundle Readiness Summary

## Goal

Expose the M028 `main_pipeline_preview.summary` integration bundle readiness
fields in `DirectorRuntime` so the runtime guard panel can show the same
offline integration blocker without reading nested registry internals.

## Scope

- Frontend adapter only.
- Offline/control-plane only.
- No WING contact.
- No backend contact beyond consuming an already loaded preview payload.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M029-CP01 Director Runtime Guard Summary

Update `frontend/src/services/operatorAdapters.ts` so
`adaptDirectorRuntime(...)` reads the
`autonomy_promotion_offline_integration_bundle_*` fields from
`pipelinePreview.summary` and surfaces them as:

- a read-only `statusItems` row labeled `Integration bundle`;
- explicit `blockedReasons` tags for status, blockers, missing checks and
  authority-denial flags while the bundle is not ready.

### M029-CP02 Contract Test and Evidence Sync

Add a frontend contract test for the Director runtime guard summary, update
roadmap, completion gate, capability matrix, current state and autopilot
evidence so M029 is durable and M001 remains
`WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- `DirectorRuntime.statusItems` includes an integration bundle readiness row
  when preview summary fields are present.
- `DirectorRuntime.blockedReasons` includes integration bundle status,
  blockers, missing checks and authority-denial tags when the bundle is not
  ready.
- The Director runtime surface remains read-only.
- Passing or blocked frontend-visible bundle status cannot grant live
  authority, readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m029_director_runtime_surfaces_integration_bundle_readiness_summary -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M029 Director runtime adapter/test and accompanying docs/state
evidence. The M026 backend readiness overlay, M027 module-status surface and
M028 preview summary remain intact.
