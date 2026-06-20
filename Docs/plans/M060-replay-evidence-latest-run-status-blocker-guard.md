# M060 Replay Evidence Latest Run Status Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. The operator UI already surfaces integration-bundle
blockers across Task Queue, Supervised Apply, Operator Decision, Safety Monitor
and Replay Evidence trace status.

The Replay Evidence `Latest run` badge can still show the raw replay status
while promotion planning is globally blocked. A passed offline replay run must
not look like live/promotion readiness.

## Scope

- Add a local blocked status derivation for Replay Evidence `Latest run`.
- Show a red `bundle blocked` badge while `integrationBundleApplyBlocked` is
  true.
- Preserve the normal replay status and tone when the blocker is absent.
- Add a frontend contract assertion and a WING MVP rail for the visible guard.
- Update roadmap, capability matrix, completion gate, current state, heartbeat,
  last report and run ledger.

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No queue mutation.
- No task data mutation.
- No replay run input mutation.
- No decision trace input mutation.
- No safety monitor input mutation.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M060-CP01: Replay Evidence `Latest run` derives local status from
  `integrationBundleApplyBlocked`.
- M060-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m060_replay_evidence_latest_run_status_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
