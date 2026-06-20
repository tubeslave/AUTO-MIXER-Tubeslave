# M074 Supervised Apply Parameter Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Supervised Apply already shows integration-bundle
blocker notices, disabled confirm controls, no-contact gate rows, blocked
readiness, hidden backend results, blocked new values and no-live-apply
rollback values while offline integration-bundle blockers are visible.

The Supervised Apply `Proposal Target` still renders the raw `Parameter`
proposal text. A raw parameter value can look like a concrete console parameter
is eligible for apply inside the modal even while global promotion planning is
blocked and no live apply may be dispatched.

## Scope

- Reuse the local blocked parameter display derivation for Supervised Apply
  `Parameter`.
- Show `Blocked parameter` with danger styling while
  `integrationBundleApplyBlocked` is true.
- Preserve the normal proposal parameter text when the blocker is absent.
- Add a frontend contract assertion and a WING MVP rail for the visible guard.
- Update roadmap, capability matrix, completion gate, current state, heartbeat,
  last report and run ledger.

## Out of Scope

- No backend contact.
- No WING contact.
- No OSC or websocket transport.
- No queue mutation.
- No task data mutation.
- No proposal data mutation.
- No task status input mutation.
- No backend result mutation.
- No row hiding.
- No supervised apply opening.
- No dispatch of supervised apply.
- No readiness closure.
- No live authority grant.
- No Full Autonomous readiness claim.

## Checkpoints

- M074-CP01: Supervised Apply derives the Proposal Target `Parameter` display
  from `taskProposalParameterDisplay(proposal, integrationBundleApplyBlocked)`.
- M074-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m074_supervised_apply_parameter_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
