# ADR: Director Event Integration Bundle Blocker Summary

## Context

M029 added integration bundle readiness tags to `DirectorRuntime.statusItems`
and `DirectorRuntime.blockedReasons`. The Director event model still rendered
generic blocker messages from raw blocked-reason tags, so an operator could see
machine-oriented `integration_bundle_*` strings instead of a readable event.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only event display only.

## Options Considered

1. Keep raw integration bundle tags in the generic `Blockers` event.
2. Convert integration bundle tags into one readable Director event.
3. Add a new backend event type.

## Decision

Convert `integration_bundle_*` blocked-reason tags into one read-only
`Integration bundle` Director event in `directorEventRows(...)`.

The event reports status, blocker count, missing checks and authority-denial
flags. Generic non-integration blockers still use the existing `Blockers`
event. This does not queue proposals, contact backend/WING, send transport,
grant readiness closure or grant live authority.

## Why This Won

The frontend already has all required runtime tags from M029. Formatting them
in the Director event model makes the cockpit easier to read without changing
backend contracts or creating a new source of authority.

## Rejected Alternatives

Leaving raw tags was rejected because it weakens operator understanding during
review.

Adding a backend event type was rejected because this is a display formatting
issue and the existing read-only runtime contract is sufficient.

## Implementation Plan

- Add an integration bundle event formatter in
  `frontend/src/components/dashboard/directorPanelModel.ts`.
- Keep generic blocker rendering for non-integration blockers.
- Cover the behavior in
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m030_director_events_render_integration_bundle_blocker_summary -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: event rows could hide raw diagnostic tags that are useful to developers.
This is acceptable because the tags remain in `DirectorRuntime.blockedReasons`
and tests assert only the operator event formatting.

Rollback: revert the M030 frontend model/test/docs/state changes. The
M026-M029 readiness surfaces remain available.
