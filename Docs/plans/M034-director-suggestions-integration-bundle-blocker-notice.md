# M034 Director Suggestions Integration Bundle Blocker Notice

## Goal

Make the Director Agent Suggestions panel show the offline integration bundle
blocker beside review-only suggestions, so suggestions are not mistaken for
promotion-ready actions when promotion planning is blocked.

## Scope

- Frontend Director panel only.
- Offline/control-plane display only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M034-CP01 Agent Suggestions Blocker Notice

Update `frontend/src/views/DirectorPanel.tsx` so visible Agent Suggestions show
a read-only `Promotion planning blocked. Suggestions remain review-only`
notice when Director runtime blocked reasons include offline integration bundle
blocker evidence.

### M034-CP02 Contract Test and Evidence Sync

Extend the frontend operator adapter contract test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M034 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- The Agent Suggestions panel renders the blocker notice only when suggestions
  are present and Director blocked-reason messages exist.
- Suggestions remain review-only and no accept/apply path is added.
- No queue mutation is introduced by the notice.
- Notice-visible integration bundle status cannot grant live authority,
  readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m034_director_suggestions_show_integration_bundle_blocker_notice -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M034 Director panel/test and accompanying docs/state evidence. The
M026-M033 readiness, event, banner, goal and queue badge surfaces remain
intact.
