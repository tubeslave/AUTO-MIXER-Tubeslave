# M059 Replay Evidence Decision Trace Status Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: a real WING Rack read-only
artifact and operator review are still required before physical readback or any
supervised live proof. Recent frontend guards make Task Queue, Supervised Apply,
Operator Decision and Safety Monitor surfaces fail closed when the offline
integration bundle blocks promotion planning.

Replay Evidence still renders the latest decision trace summary. Without a
local blocker-derived status, an operator can see trace details without an
adjacent indication that promotion planning is globally blocked.

## Scope

- Add a local `integrationBundleApplyBlocked` input to `ReplayEvidencePanel`.
- Show a red `bundle blocked` status badge for the Trace row while the blocker
  is visible.
- Preserve the normal trace `blocked` or `proofStatus` status when the blocker
  is absent.
- Add a frontend contract assertion and a WING MVP rail for the visible guard.
- Update roadmap, capability matrix, completion gate, current state, heartbeat,
  last report and run ledger.

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No queue mutation.
- No task data mutation.
- No decision trace input mutation.
- No safety monitor input mutation.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M059-CP01: Replay Evidence Trace row derives local status from
  `integrationBundleApplyBlocked`.
- M059-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m059_replay_evidence_decision_trace_status_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
