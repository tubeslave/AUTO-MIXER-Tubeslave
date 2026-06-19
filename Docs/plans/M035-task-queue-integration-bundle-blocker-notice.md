# M035 Task Queue Integration Bundle Blocker Notice

## Goal

Make the Task Queue show the offline integration bundle blocker before proposal
actions, so operator review of queued tasks cannot be mistaken for promotion
planning or live-apply readiness.

## Scope

- Frontend Task Queue only.
- Offline/control-plane display only.
- No WING contact.
- No backend contact.
- No OSC or websocket transport.
- No live write.
- No queue mutation.
- No readiness closure, Full Autonomous eligibility, hardware evidence or live
  authority grant.

## Checkpoints

### M035-CP01 Task Queue Blocker Notice

Update `frontend/src/components/prototype/TaskQueuePanels.tsx` and
`frontend/src/views/TaskQueueView.tsx` so Task Queue receives Director runtime
blocked reasons and renders a read-only promotion-planning blocker notice above
proposal actions.

### M035-CP02 Contract Test and Evidence Sync

Extend the frontend operator adapter contract test, update roadmap, completion
gate, capability matrix, current state and autopilot evidence so M035 is
durable and M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`.

## Acceptance Criteria

- Task Queue renders a blocker notice when task rows and Director
  blocked-reason messages are both visible.
- Proposal rows remain inspectable and review-only for integration bundle
  evidence.
- No queue mutation is introduced by the notice.
- Queue-notice-visible integration bundle status cannot grant live authority,
  readiness closure, hardware evidence or Full Autonomous readiness.
- M001 still waits for a real WING read-only artifact and operator review.

## Validation

```bash
npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs
npm --prefix frontend run build
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m035_task_queue_shows_integration_bundle_blocker_notice -q
PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q
PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q
```

## Rollback

Revert the M035 Task Queue panel/view/test and accompanying docs/state
evidence. The M026-M034 readiness, Director and Agent Suggestions surfaces
remain intact.
