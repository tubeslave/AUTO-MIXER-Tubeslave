# M063 Replay Evidence Report Path Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Replay Evidence now guards the latest-run badge, trace
status, policy row and channel status when offline integration-bundle blockers
are visible.

The Replay Evidence report line still renders the raw replay `reportPath`. That
path can point at stale or review-only replay evidence and may look usable while
the global promotion-planning blocker is active.

## Scope

- Add local blocked report-path derivation for Replay Evidence.
- Show `bundle blocked` with danger styling while `integrationBundleApplyBlocked`
  is true.
- Preserve the normal replay report path or `pending` text when the blocker is
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
- No replay report input mutation.
- No channel evidence input mutation.
- No replay run input mutation.
- No decision trace input mutation.
- No safety monitor input mutation.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M063-CP01: Replay Evidence report path text derives local status from
  `integrationBundleApplyBlocked`.
- M063-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m063_replay_evidence_report_path_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
