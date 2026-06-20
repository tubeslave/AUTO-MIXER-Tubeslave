# M076 Replay Evidence Start Action Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Replay Evidence panel status rows already show blocked
integration-bundle state while offline integration-bundle blockers are visible.

The Replay Evidence panel still exposes the `Start Offline` action directly to
`onStartReplay`. In the current frontend/backend wiring, that can ask the
Product Layer websocket to start a replay run. While promotion planning is
blocked, the blocker surface should remain read-only and must not initiate a
backend replay request from the UI.

## Scope

- Derive a local Replay Evidence start-action blocked state from
  `integrationBundleApplyBlocked`.
- Render `Replay Blocked` and disable the action while blockers are visible.
- Fail closed in the click handler before `onStartReplay`.
- Preserve `Start Offline` behavior when the blocker is absent.
- Add a frontend contract assertion and a WING MVP rail for the visible guard.
- Update roadmap, capability matrix, completion gate, current state, heartbeat,
  last report and run ledger.

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport while the blocker is visible.
- No replay run input mutation.
- No decision trace input mutation.
- No channel evidence input mutation.
- No task data mutation.
- No queue mutation.
- No proposal creation.
- No supervised apply opening.
- No dispatch of supervised apply.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M076-CP01: Replay Evidence derives local start-action label, class and
  disabled state from `replayStartBlocked`.
- M076-CP02: Frontend and WING rails prove the guard returns before
  `onStartReplay` while integration-bundle blockers are visible and cannot
  grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m076_replay_evidence_start_action_is_blocked_by_integration_bundle -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
