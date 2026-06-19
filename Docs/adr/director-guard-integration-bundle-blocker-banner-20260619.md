# ADR: Director Guard Integration Bundle Blocker Banner

## Context

M030 made Director events render `integration_bundle_*` blocked-reason tags as a
readable event. The Runtime Guard banner in `RightDirectorPanel` still rendered
the first raw `runtime.blockedReasons` values directly, so an operator could
still see machine-oriented integration bundle tags in the guard panel.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only banner display only.

## Options Considered

1. Leave the Runtime Guard banner rendering raw blocked-reason tags.
2. Share the Director blocked-reason formatter between event rows and the guard
   banner.
3. Remove the guard blocker banner entirely.

## Decision

Share a read-only Director blocked-reason formatter and use it in the Runtime
Guard banner.

The formatter converts integration bundle tags into one readable message while
preserving non-integration blockers. This does not queue proposals, contact
backend/WING, send transport, grant readiness closure or grant live authority.

## Why This Won

The runtime guard should explain blockers in operator-facing language. Reusing
the existing Director formatting logic keeps event and banner wording
consistent without changing backend contracts or authority boundaries.

## Rejected Alternatives

Leaving raw tags was rejected because it weakens operator understanding in the
main runtime guard.

Removing the banner was rejected because non-integration blockers still need a
compact visible surface.

## Implementation Plan

- Export integration bundle blocked-reason formatting from
  `frontend/src/components/dashboard/directorPanelModel.ts`.
- Use the formatter in `frontend/src/views/DirectorPanel.tsx`.
- Cover the behavior in
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m031_director_guard_banner_renders_integration_bundle_blocker_summary -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: developer-oriented raw tags become less visible in the banner. This is
acceptable because raw tags remain in `DirectorRuntime.blockedReasons` and the
contract test asserts operator-readable formatting only.

Rollback: revert the M031 frontend model/panel/test/docs/state changes. The
M026-M030 readiness and event surfaces remain available.
