# ADR: Director Queue Integration Bundle Blocker Badge

## Context

M029-M032 made the Director runtime guard, event rows, Runtime Guard banner and
current-goal text surface offline integration bundle blockers as read-only
operator evidence. The same blocked preview could still leave the proposal
queue badge green as `Proposal queue online` when backend queue state was
available, even though promotion planning remained blocked and queue mutation
was denied.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only badge display only.

## Options Considered

1. Leave the proposal queue badge tied only to backend queue availability.
2. Override the badge with a red promotion-planning blocker state when the
   offline integration bundle is blocked.
3. Hide all proposal rows whenever the offline integration bundle is blocked.

## Decision

Override the proposal queue badge with `Promotion planning blocked` and red
tone when `DirectorRuntime.blockedReasons` includes integration bundle blocker
evidence.

This does not queue proposals, hide review-only task rows, contact
backend/WING, send transport, grant readiness closure or grant live authority.

## Why This Won

Backend queue availability and promotion-planning readiness are different
signals. Keeping a green queue badge beside a blocked integration bundle can
make the operator think promotion planning is available. A blocker-aware badge
keeps the UI truthful while preserving review-only evidence rows.

## Rejected Alternatives

Leaving the badge unchanged was rejected because it weakens the M029-M032
operator-facing blocker surface.

Hiding proposal rows was rejected because review-only evidence can still be
useful to inspect and should not be silently removed.

## Implementation Plan

- Add a blocker-aware proposal queue label/tone in
  `frontend/src/services/operatorAdapters.ts`.
- Extend
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to assert
  the label and tone for blocked integration bundle preview data.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m033_director_queue_badge_surfaces_integration_bundle_blocker -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: the badge may read as if all backend proposal inspection is blocked. The
wording uses `Promotion planning blocked` rather than `Queue unavailable`, and
review-only rows remain visible.

Rollback: revert the M033 frontend adapter/test/docs/state changes. The
M026-M032 readiness, event, banner and goal surfaces remain available.
