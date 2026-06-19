# ADR: Task Details Apply Entrypoint Blocker Guard

## Context

M039 guarded the shared `requestTaskApply` helper, but `TaskDetailsPanel` still
called that helper without the integration bundle blocked state. That detail
panel button is a separate operator-facing entrypoint from the table action
cluster, so it must receive the same blocked state before it can open
Supervised Apply.

## Options Considered

1. Rely on the helper default argument and the hidden table Apply action.
2. Pass the integration bundle blocked state into `TaskDetailsPanel` and keep
   the detail apply button blocked.
3. Add backend mutation changes in this checkpoint.

## Decision

Compute the integration bundle blocked state in `TaskQueueView`, pass it into
`TaskDetailsPanel`, include it in `applyBlocked`, and pass it through to
`requestTaskApply` for any future detail button wiring.

This does not open Supervised Apply, queue proposals, dispatch supervised
apply, contact backend/WING, grant live authority or close readiness.

## Why This Won

The detail panel is the remaining visible Task Queue apply entrypoint outside
the table row action cluster. Passing the same blocked state keeps the frontend
contract consistent and protects against future UI refactors without changing
backend safety gates.

## Rejected Alternatives

Relying on the helper default argument was insufficient because the default is
intentionally unblocked for callers without integration context. Backend
mutation changes were rejected because this checkpoint only hardens a local
frontend entrypoint and cannot claim backend or hardware proof.

## Implementation Plan

1. Import `directorBlockedReasonMessages` in `TaskQueueView`.
2. Compute `integrationBundleApplyBlocked` from Director runtime blockers.
3. Pass the flag into `TaskDetailsPanel`.
4. Include the flag in `applyBlocked` and pass it to `requestTaskApply`.
5. Add frontend contract and WING rail coverage.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m040_task_details_apply_entrypoint_is_guarded_by_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Details apply-button behavior. If it
regresses, revert the prop wiring and M040 evidence/test/docs changes. The
rollback does not alter backend gates, WING transport or accepted hardware
evidence.
