# ADR: Supervised Apply Readiness Blocker Guard

## Context

M048 makes the Task Queue `Operator Decision` panel show blocked write-path and
backend statuses while offline integration bundle blockers are visible. The
Supervised Apply modal still rendered `Final readiness` directly from
`ui.backend.statusLabel` and `ui.backend.statusTone`, so a blocked modal could
show a ready-looking backend readiness row beside disabled apply controls and a
blocked gate sequence.

## Options Considered

1. Leave the modal final-readiness row unchanged and rely on the disabled apply
   button.
2. Derive a local blocked final-readiness status while
   `integrationBundleApplyBlocked` is true.
3. Mutate the backend status input or backend proposal state.

## Decision

Add local `finalReadinessStatus` derivation inside `SupervisedApplyModal`.
When the global blocker is absent, the modal renders the existing backend
status values unchanged. When the blocker is visible, the `Final readiness`
row shows `Blocked by integration bundle` with red tone.

This does not mutate backend status input, task data, queue proposals, open
Supervised Apply, dispatch supervised apply, contact backend/WING, grant live
authority or close readiness.

## Why This Won

The modal is the highest-friction apply review surface. Its readiness row must
match the disabled confirm button and blocked gate sequence so the operator
does not see a readiness signal while promotion planning is blocked.

## Rejected Alternatives

Leaving the row unchanged would preserve a misleading ready-looking status in
the apply review surface. Mutating backend input was rejected because this
checkpoint is frontend presentation only and cannot claim backend, hardware or
operator proof.

## Implementation Plan

1. Add local `finalReadinessStatus` derivation in `SupervisedApplyModal`.
2. Render the `Final readiness` row from that derived status.
3. Preserve the existing `disabled={!canRequestBackendGate}` apply control.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m049_supervised_apply_final_readiness_reflects_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Supervised Apply readiness presentation. If it
regresses, revert the local derived status and M049 evidence/test/docs updates.
The rollback does not alter backend gates, WING transport or accepted hardware
evidence.
