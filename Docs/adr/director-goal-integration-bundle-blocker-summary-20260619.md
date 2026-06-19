# ADR: Director Goal Integration Bundle Blocker Summary

## Context

M029-M031 made the Director runtime guard, events and Runtime Guard banner
surface offline integration bundle readiness and blockers as read-only
operator evidence. The `DirectorRuntime.currentGoal` text still fell through
to the generic evidence-collection fallback when the same preview reported a
blocked offline integration bundle.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only goal text only.

## Options Considered

1. Leave `currentGoal` generic while status rows and banners show the blocker.
2. Prioritize the integration bundle blocker in `currentGoal` after emergency
   stop and before the supervised-gate fallback.
3. Make `currentGoal` include every raw blocked-reason tag.

## Decision

Prioritize the offline integration bundle blocker in `currentGoal` after
emergency stop. The goal text tells the operator to resolve the offline
integration bundle blocker before promotion planning and states that proposals
remain review-only with no live authority, readiness closure or console apply.

This does not queue proposals, contact backend/WING, send transport, grant
readiness closure or grant live authority.

## Why This Won

The Director goal should name the highest relevant non-emergency blocker in
operator-facing language. Keeping emergency stop first preserves the existing
safety latch priority, while a concise blocker goal avoids leaking raw machine
tags into the primary goal text.

## Rejected Alternatives

Leaving the generic goal was rejected because it hides the active offline
integration bundle blocker from the most prominent Director text.

Listing every raw tag was rejected because status rows, events and the Runtime
Guard banner already expose the detailed blocker evidence.

## Implementation Plan

- Add an integration bundle blocker goal in
  `frontend/src/services/operatorAdapters.ts`.
- Extend
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to assert
  the `currentGoal` wording for a blocked integration bundle preview.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m032_director_goal_surfaces_integration_bundle_blocker_summary -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: `currentGoal` may show the integration bundle blocker even when the
supervised gate is otherwise available. This is acceptable because the blocker
concerns promotion planning and the text explicitly denies console apply.

Rollback: revert the M032 frontend adapter/test/docs/state changes. The
M026-M031 readiness, event and banner surfaces remain available.
