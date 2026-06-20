# M072 Task Details Proposal Proof Blocker Guard

## Context

M001 remains `WAITING_FOR_EXTERNAL_VALIDATION`: real WING Rack read-only
evidence and operator review are still required before physical readback or
supervised live proof. Task Details already shows blocked status, blocked risk,
blocked delta, blocked rollback, blocker lists and disabled apply/reject
controls while offline integration-bundle blockers are visible.

The Task Details `Proposal Preview` still renders the raw `Proof` proposal text.
A raw proof value can look like available live/supervised proof while global
promotion planning is blocked and no live apply has occurred.

## Scope

- Add local blocked proof display derivation for Task Details `Proof`.
- Show `No live proof` with danger styling while
  `integrationBundleApplyBlocked` is true.
- Preserve the normal proposal proof text when the blocker is absent.
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

- M072-CP01: Task Details derives the Proposal Preview `Proof` display from
  `taskProposalProofDisplay(proposal, integrationBundleApplyBlocked)`.
- M072-CP02: Frontend and WING rails prove the guard remains presentation-only
  and cannot grant live authority.

## Validation

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m072_task_details_proposal_proof_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`
