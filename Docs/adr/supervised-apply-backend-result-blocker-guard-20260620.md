# ADR: Supervised Apply Backend Result Blocker Guard

## Context

M049 makes the Supervised Apply modal `Final readiness` row show blocked status
while offline integration bundle blockers are visible. The same modal still
used a matching `ui.backend.proposalApplyResult` for apply request, readback,
rollback and the `Backend Result` panel. A stale or prior backend result could
therefore look like current active apply evidence inside a modal that is
globally blocked by promotion-planning evidence.

## Options Considered

1. Leave matched backend result presentation unchanged and rely on the blocked
   final-readiness row.
2. Keep the backend result input untouched but hide it as active modal evidence
   while `integrationBundleApplyBlocked` is true.
3. Mutate or clear the backend result input when the integration bundle blocker
   is visible.

## Decision

Add local `visibleBackendResultMatches` derivation inside
`SupervisedApplyModal`. Matching backend result data is allowed to drive modal
rows only when the integration bundle blocker is absent. While the blocker is
visible, the modal falls back to blocked/current review labels and does not
render the `Backend Result` panel.

This does not mutate backend result input, backend status input, task data,
queue proposals, open Supervised Apply, dispatch supervised apply, contact
backend/WING, grant live authority or close readiness.

## Why This Won

This keeps stale or prior backend result evidence from reading like current
apply readiness in the blocked modal, while preserving the underlying backend
state and deterministic gates unchanged.

## Rejected Alternatives

Leaving the result presentation unchanged would keep a ready-looking backend
result beside blocked controls. Clearing backend input was rejected because this
checkpoint is frontend presentation only and cannot claim backend, hardware or
operator proof.

## Implementation Plan

1. Add `visibleBackendResultMatches` in `SupervisedApplyModal`.
2. Use that guard for apply request, readback, rollback and backend-result
   tone derivation.
3. Render the `Backend Result` panel only when the guarded match is visible.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m050_supervised_apply_backend_result_is_hidden_by_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Supervised Apply backend-result presentation. If it
regresses, revert the local visible-result guard and M050 evidence/test/docs
updates. The rollback does not alter backend gates, WING transport or accepted
hardware evidence.
