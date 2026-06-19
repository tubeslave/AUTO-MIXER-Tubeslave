# ADR: Operator report contract

## Context

Offline integration evidence must be understandable to the operator before it
can support review, promotion or later hardware handoff. The integration bundle
still has a loose `operator_report_bound` check, so the repo needs a concrete
operator-report contract before that check can be safely bound.

## Options Considered

1. Keep `operator_report_bound` as a loose integration check.
2. Bind operator reports directly into the integration bundle without a
   standalone contract.
3. Add a standalone operator-report contract first, then bind it in a later
   milestone.

## Decision

Add `backend/operator_report_contract.py` as a standalone fail-closed evaluator
for offline operator-facing report manifests.

## Why This Won

The standalone contract gives us focused tests for report completeness,
authority denial, evidence bindings, visible blockers, next operator action and
audit fields before changing integration bundle promotion semantics. The
evaluator always returns `live_authority_granted: false`.

## Rejected Alternatives

Keeping a loose boolean does not prove the report is actionable or safe to show
as review evidence. Binding immediately would mix two changes: defining the
contract and changing integration promotion behavior.

## Implementation Plan

- Add the operator report evaluator and unit tests.
- Add a WING MVP rail that checks the roadmap, completion gate, current state,
  ADR and report wording.
- Update autopilot state, heartbeat and ledger evidence.

## Test Plan

- Run `tests/test_operator_report_contract.py`.
- Run the M024 WING rail and full WING rail suite.
- Run the standard full pytest command before commit.

## Risks and Rollback

The main risk is over-claiming operator-report evidence as readiness or hardware
proof. The evaluator explicitly denies those authority paths. Rollback is
reverting this ADR's commit; no live console state is created or changed.
