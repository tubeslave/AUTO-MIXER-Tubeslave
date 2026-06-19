# ADR: Director Suggestions Integration Bundle Blocker Notice

## Context

M029-M033 made the Director runtime guard, event rows, Runtime Guard banner,
current-goal text and proposal queue badge surface offline integration bundle
blockers as read-only operator evidence. The Agent Suggestions panel could
still show review-only suggestions without a local warning inside that panel,
which could make suggestions look closer to promotion-ready actions than they
are.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only panel display only.

## Options Considered

1. Leave the Agent Suggestions panel unchanged and rely on the Runtime Guard
   banner and queue badge.
2. Show a blocker notice inside Agent Suggestions while keeping suggestions
   visible and review-only.
3. Hide suggestions whenever the offline integration bundle is blocked.

## Decision

Show a read-only Agent Suggestions notice:
`Promotion planning blocked. Suggestions remain review-only` when Director
runtime blocked-reason messages exist and suggestions are visible.

This does not queue proposals, hide review-only suggestion rows, contact
backend/WING, send transport, grant readiness closure or grant live authority.

## Why This Won

The blocker should be visible at the point where the operator reviews
suggestions. A local notice keeps the panel truthful without removing evidence
that may still be useful to inspect.

## Rejected Alternatives

Leaving the panel unchanged was rejected because it relies on the operator
cross-reading other Director surfaces before reviewing suggestions.

Hiding suggestions was rejected because review-only rows remain useful
evidence and hiding them can make the system look less auditable.

## Implementation Plan

- Add a blocker notice to `frontend/src/views/DirectorPanel.tsx`.
- Extend
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to assert
  the notice wiring in the active right Director panel source.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m034_director_suggestions_show_integration_bundle_blocker_notice -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: the notice could make all suggestion review look disabled. The wording
says suggestions remain review-only and the existing Review/Reject controls
continue to send local toasts only.

Rollback: revert the M034 frontend panel/test/docs/state changes. The
M026-M033 readiness, event, banner, goal and queue badge surfaces remain
available.
