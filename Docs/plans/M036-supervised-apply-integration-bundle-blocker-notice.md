# M036 Supervised Apply Integration Bundle Blocker Notice

## Goal

Make the Supervised Apply modal show the offline integration bundle blocker
before supervised apply review, so an operator cannot mistake local proposal
inspection for promotion planning or live-apply readiness.

## Scope

- Frontend Supervised Apply modal only.
- Offline/control-plane display only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M036-CP01 Supervised Apply Blocker Notice

Update `frontend/src/components/prototype/TaskQueuePanels.tsx` so
`SupervisedApplyModal` receives Director runtime blocked reasons and renders a
read-only promotion-planning blocker notice before supervised apply review.

### M036-CP02 Contract Test and Evidence Sync

Extend the frontend operator adapter contract test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M036 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Supervised Apply modal renders a blocker notice when a modal task and
  Director blocked-reason messages are both visible.
- Existing apply gate labels remain governed by their current UI/backend
  status.
- No queue mutation is introduced by the notice.
- Modal-notice-visible integration bundle status cannot grant live authority,
  readiness closure, hardware evidence, bypass apply gates or Full Autonomous
  readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m036_supervised_apply_modal_shows_integration_bundle_blocker_notice -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M036 Supervised Apply modal/test and accompanying docs/state
evidence. The M026-M035 readiness, Director, Agent Suggestions and Task Queue
surfaces remain intact.
