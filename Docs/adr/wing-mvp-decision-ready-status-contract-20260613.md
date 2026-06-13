# ADR: WING MVP decision-ready status contract

## 1. Context

The supervised WING MVP cannot progress to fader/gain proof until a real
read-only preflight candidate has been manually accepted. The handoff/proof
contract already requires the decision-review status `accepted_decision_ready`,
but the decision-review CLI also carried that status as a local string.

Local ownership creates a drift path: the CLI can report one ready status while
the supervised prestart gate requires another.

## 2. Options considered

- Keep the CLI-local status literal.
- Bind the CLI ready status to the shared WING preflight contract.
- Centralize all decision-review statuses now.

## 3. Decision

Bind the CLI's accepted decision-ready status to
`WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_DECISION_STATUS`.

## 4. Why this won

This is the smallest change that protects the transition from operator review
to supervised proof planning. It does not widen live authority, does not contact
hardware, and keeps non-gating statuses local.

## 5. Rejected alternatives

Keeping the local literal leaves the drift path open. Centralizing every status
is broader than needed for the current WING MVP gate.

## 6. Implementation plan

1. Add a decision-review CLI alias sourced from the shared contract.
2. Use that alias for accepted-ready report status and exit-code selection.
3. Add tests comparing the CLI status with the supervised prestart gate
   required decision status.

## 7. Test plan

- Focused decision-review/proof-plan tests.
- WING MVP rails tests.
- Full `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.

## 8. Risks and rollback

Risk is limited to report status selection for already accepted decision
reviews. Rollback is to restore the CLI-local literal, but that would re-open
the drift risk.
