# ADR: Director Runtime Integration Bundle Readiness Summary

## Context

M026 added an `offline_integration_bundle` overlay to autonomy promotion
readiness. M027 surfaced that overlay in operator-facing module status rows.
M028 copied the same readiness summary into `main_pipeline_preview.summary`.

`DirectorRuntime` powers the runtime guard panel and current Director context,
but it did not yet show the M028 integration bundle blocker. An operator could
therefore see preview and module-status blockers while the Director guard
continued to report only generic preview status.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only runtime display only.

## Options Considered

1. Leave integration bundle readiness out of `DirectorRuntime`.
2. Add a dedicated `DirectorRuntime` field and update all UI consumers.
3. Reuse existing `statusItems` and `blockedReasons` surfaces in
   `DirectorRuntime`.

## Decision

Reuse existing `DirectorRuntime.statusItems` and `DirectorRuntime.blockedReasons`
to surface integration bundle readiness from `pipelinePreview.summary`.

The adapter adds a read-only `Integration bundle` status row and explicit
blocked-reason tags for bundle status, blockers, missing checks and
authority-denial flags when the bundle is not ready. This does not add a live apply path,
queue action, backend contact, WING contact, readiness closure or live authority
grant.

## Why This Won

`statusItems` and `blockedReasons` are already rendered by the runtime guard
panel. Reusing them keeps the contract stable, avoids UI churn and makes the
promotion blocker visible in the same area operators already use for runtime
guard state.

## Rejected Alternatives

Leaving the runtime guard unchanged was rejected because it would hide a
promotion-critical blocker from the Director surface.

Adding new type fields was rejected for now because the existing status and
blocker lists carry the required read-only evidence without broad frontend
changes.

## Implementation Plan

- Read `autonomy_promotion_offline_integration_bundle_*` fields from
  `pipelinePreview.summary` inside `adaptDirectorRuntime(...)`.
- Add a read-only `Integration bundle` row to `statusItems` when summary
  fields are present.
- Add blocked-reason tags for bundle status, blockers, missing checks and
  authority-denial flags while readiness is not true.
- Cover the behavior in
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m029_director_runtime_surfaces_integration_bundle_readiness_summary -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: runtime guard status grows noisier. This is acceptable because the row is
present only when preview summary fields exist and the blocker is
promotion-critical.

Rollback: revert the M029 frontend adapter/test/docs/state changes. The M026,
M027 and M028 readiness surfaces remain available.
