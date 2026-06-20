# M080 Replay Evidence Report Row Blocker Visibility Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Replay Evidence already blocks report path text when the
report row is visible.

The report row is still rendered only when `latestRun` exists. While the global
blocker is visible and no replay run exists, that hides the blocked report
status instead of making the missing report proof explicit to the operator.

## Scope

- Derive a local Replay Evidence report-row visibility flag from
  `integrationBundleApplyBlocked` and `latestRun`.
- Keep the report row visible while blockers are visible.
- Preserve the existing `latestRun`-based visibility when the blocker is absent.
- Add a frontend contract assertion and a WING MVP rail for the visible guard.
- Update roadmap, capability matrix, completion gate, current state, heartbeat,
  last report and run ledger.

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No replay report input mutation.
- No replay run input mutation.
- No decision trace input mutation.
- No channel evidence input mutation.
- No task data mutation.
- No queue mutation.
- No replay start.
- No supervised apply opening.
- No dispatch of supervised apply.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M080-CP01: Replay Evidence derives report-row visibility from
  `integrationBundleApplyBlocked || Boolean(latestRun)`.
- M080-CP02: Frontend and WING rails prove the report-row visibility guard
  remains presentation-only and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m080_replay_evidence_report_row_visible_under_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
