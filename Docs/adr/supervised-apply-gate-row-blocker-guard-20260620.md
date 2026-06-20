# ADR: Supervised Apply Gate Row Blocker Guard

## Context

M050 hides matched backend results while offline integration bundle blockers are
visible. The Supervised Apply modal still fell back to normal gate-row wording
for readback and rollback: `Required after backend apply` and `Snapshot
required`. That wording could still look like a pending backend workflow inside
a modal that is globally blocked by promotion-planning evidence.

## Options Considered

1. Leave readback and rollback rows unchanged and rely on the blocked final
   readiness row.
2. Derive local blocked Apply Gate row labels while
   `integrationBundleApplyBlocked` is true.
3. Mutate backend result or backend status input to force the modal state.

## Decision

Add local `modalGateBlockerStatus` derivation inside `SupervisedApplyModal`.
When the global blocker is visible, the modal shows blocked apply request, no
backend contact for readback and no live apply for rollback, with red tones.
When the blocker is absent, the existing backend-result/readback/rollback
presentation remains unchanged.

This does not mutate backend result input, backend status input, task data,
queue proposals, open Supervised Apply, dispatch supervised apply, contact
backend/WING, grant live authority or close readiness.

## Why This Won

The Apply Gate rows are the operator's step-by-step mental model for the
supervised apply path. In a globally blocked state, those rows should describe
the blocked path directly rather than imply a backend apply/readback/rollback
workflow can continue.

## Rejected Alternatives

Leaving the rows unchanged would keep workflow language that conflicts with the
blocked modal state. Mutating backend input was rejected because this checkpoint
is frontend presentation only and cannot claim backend, hardware or operator
proof.

## Implementation Plan

1. Add `modalGateBlockerStatus` in `SupervisedApplyModal`.
2. Use it for apply request, readback, rollback labels and tones when blocked.
3. Preserve existing backend-result behavior when the blocker is absent.
4. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m051_supervised_apply_gate_rows_reflect_integration_bundle_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Supervised Apply gate-row presentation. If it
regresses, revert the local gate-row status derivation and M051
evidence/test/docs updates. The rollback does not alter backend gates, WING
transport or accepted hardware evidence.
