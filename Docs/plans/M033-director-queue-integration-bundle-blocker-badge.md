# M033 Director Queue Integration Bundle Blocker Badge

## Goal

Make the Director runtime proposal-queue badge reflect the offline integration
bundle blocker when promotion planning is blocked, so the Runtime Guard does
not show a green `Proposal queue online` state beside a blocked integration
bundle.

## Scope

- Frontend Director runtime adapter only.
- Offline/control-plane display only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M033-CP01 Proposal Queue Badge Blocker Priority

Update `frontend/src/services/operatorAdapters.ts` so an integration bundle
blocker changes `DirectorRuntime.proposalQueueLabel` to
`Promotion planning blocked` and `proposalQueueTone` to `red`, without changing
backend queue state or task rows.

### M033-CP02 Contract Test and Evidence Sync

Extend the frontend operator adapter contract test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M033 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Blocked offline integration bundle preview data changes the Director proposal
  queue badge to a blocker-focused label.
- Backend proposal queue availability is not treated as promotion readiness.
- Review-only task rows remain review-only and no queue mutation is introduced.
- Badge-visible integration bundle status cannot grant live authority,
  readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m033_director_queue_badge_surfaces_integration_bundle_blocker -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M033 Director runtime adapter/test and accompanying docs/state
evidence. The M026-M032 readiness, event, banner and goal surfaces remain
intact.
