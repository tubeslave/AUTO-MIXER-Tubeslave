# ADR: Operator Decision Apply Result Blocker Guard

## Context

M048 made the Operator Decision panel show blocked write-path and no-contact
backend statuses while offline integration bundle blockers are visible. The
same panel still displayed `Apply result` from the normal operator policy,
which could look like an ordinary apply outcome even though promotion planning
is globally blocked.

## Options Considered

1. Leave `Apply result` unchanged because write path and backend rows already
   show the blocker.
2. Derive a local blocked `Apply result` presentation from
   `integrationBundleApplyBlocked`, while preserving the policy-derived value
   when the blocker is absent.
3. Hide the Operator Decision panel while promotion planning is blocked.

## Decision

Add local Operator Decision apply-result presentation state derived from
`integrationBundleApplyBlocked`. When the global blocker is visible, the panel
shows `Apply blocked` with danger styling. When the blocker is absent, the
existing `policy.apply` behavior remains unchanged.

This does not mutate backend status input, backend result input, task data,
queue proposals, open Supervised Apply, dispatch supervised apply, contact
backend/WING, grant live authority or close readiness.

## Why This Won

The Operator Decision panel is a compact decision summary. Showing a normal
apply result next to blocked write-path and no-contact backend rows creates a
mixed signal. A local blocked presentation keeps all rows aligned with the same
global safety boundary without changing backend state or operator policy input.

## Rejected Alternatives

Leaving `Apply result` unchanged preserved an ambiguous row on a blocked
decision surface. Hiding the panel was rejected because the operator still needs
the decision summary and blocker explanation while reviewing the queue.

## Implementation Plan

1. Add `applyResultStatus` in `OperatorDecisionPanel`.
2. Render `Apply blocked` only while `integrationBundleApplyBlocked` is true.
3. Preserve existing policy-derived apply result when the blocker is absent.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m055_operator_decision_apply_result_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Operator Decision row presentation. If it regresses,
revert the local apply-result presentation guard and M055 evidence/test/docs
updates. The rollback does not alter backend gates, WING transport or accepted
hardware evidence.
