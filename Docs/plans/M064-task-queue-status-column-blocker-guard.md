# M064 Task Queue Status Column Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Task Queue already shows blocker notices, blocks apply
actions and marks rows with blocked badges when offline integration-bundle
blockers are visible.

The Task Queue `Status` column still renders the raw task status. A raw
`Applied` or `Pending` badge can look like current apply readiness while global
promotion planning is blocked.

## Scope

- Add local blocked status-column derivation for Task Queue rows.
- Show `Blocked` with a red badge while `integrationBundleApplyBlocked` is true.
- Preserve the normal task status and tone when the blocker is absent.
- Add a frontend contract assertion and a WING MVP rail for the visible guard.
- Update roadmap, capability matrix, completion gate, current state, heartbeat,
  last report and run ledger.

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No queue mutation.
- No task data mutation.
- No task status input mutation.
- No row hiding.
- No supervised apply opening.
- No dispatch of supervised apply.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M064-CP01: Task Queue Status column derives local display status from
  `integrationBundleApplyBlocked`.
- M064-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m064_task_queue_status_column_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
