# ADR: Operator Decision Blocker Status Guard

## Context

M047 makes the Task Queue `Apply Plan` and Supervised Apply modal `Gate
Sequence` show blocked steps while offline integration bundle blockers are
visible. The `Operator Decision` panel still read `ui.backend.applyPath` and
`ui.backend.statusLabel` directly, so it could show a yellow write path or
normal backend status next to controls and plans that are globally blocked by
promotion-planning evidence.

## Options Considered

1. Leave the Operator Decision panel unchanged and rely on nearby blocked
   notices.
2. Derive local blocked write-path and backend statuses while
   `integrationBundleApplyBlocked` is true.
3. Mutate the backend status input or backend proposal state.

## Decision

Add local `writePathStatus` and `backendStatus` derivation inside
`OperatorDecisionPanel`. When the global blocker is absent, the panel renders
the existing backend status values unchanged. When the blocker is visible, the
panel shows `Integration bundle blocked` for the write path and `No contact`
for backend status.

This does not mutate backend status input, task data, queue proposals, open
Supervised Apply, dispatch supervised apply, contact backend/WING, grant live
authority or close readiness.

## Why This Won

Operator Decision is part of the operator-facing explanation of whether a move
can proceed. Showing blocked local statuses keeps it consistent with the
blocked apply plan, badges, filters, metrics and blocker list while preserving
backend safety gates as the real authority.

## Rejected Alternatives

Leaving the panel unchanged would keep a ready-looking write-path row visible
inside a globally blocked state. Mutating backend input was rejected because
this checkpoint is frontend presentation only and cannot claim backend,
hardware or operator proof.

## Implementation Plan

1. Add optional `integrationBundleApplyBlocked` input to
   `OperatorDecisionPanel`.
2. Derive blocked write-path and backend statuses locally when the flag is
   true.
3. Pass the existing Task Queue blocker flag into the panel.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m048_operator_decision_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Operator Decision status presentation. If it
regresses, revert the local derived statuses and M048 evidence/test/docs
updates. The rollback does not alter backend gates, WING transport or accepted
hardware evidence.
