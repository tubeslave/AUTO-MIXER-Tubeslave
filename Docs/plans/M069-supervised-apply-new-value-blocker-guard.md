# M069 Supervised Apply New Value Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Supervised Apply already shows blocker notices, blocks
confirm, hides backend results and shows no-contact/no-live gate rows while
offline integration-bundle blockers are visible.

The Supervised Apply modal `Proposal Target` panel still renders the raw
`New value` proposal delta. A raw `+1.5 dB`, trim or fader move can look like a
current apply-ready value while global promotion planning is blocked.

## Scope

- Reuse local blocked proposal-delta display derivation for Supervised Apply
  `New value`.
- Show `Blocked` with danger styling while `integrationBundleApplyBlocked` is
  true.
- Preserve the normal proposal delta when the blocker is absent.
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

- M069-CP01: Supervised Apply derives the Proposal Target `New value` display
  from `taskProposalDeltaDisplay(proposal, integrationBundleApplyBlocked)`.
- M069-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m069_supervised_apply_new_value_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
