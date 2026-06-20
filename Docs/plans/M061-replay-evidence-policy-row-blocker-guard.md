# M061 Replay Evidence Policy Row Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Replay Evidence now guards the latest-run badge and trace
status when offline integration-bundle blockers are visible.

The Replay Evidence `Policy` row still renders the raw replay policy as
`offline only` and `no live write`. That is true for replay evidence, but it
does not explicitly show the global promotion blocker at the policy row.

## Scope

- Add local blocked policy-row derivation for Replay Evidence.
- Show `bundle blocked · no live apply` while
  `integrationBundleApplyBlocked` is true.
- Preserve the normal replay policy and live-write detail when the blocker is
  absent.
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

- M061-CP01: Replay Evidence `Policy` row derives local status from
  `integrationBundleApplyBlocked`.
- M061-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m061_replay_evidence_policy_row_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
