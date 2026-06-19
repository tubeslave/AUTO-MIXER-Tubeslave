# ADR: Task Apply Request Helper Blocker Guard

## Context

M037 removed visible Apply actions and disabled Supervised Apply controls while
offline integration bundle blockers are visible. M038 also guarded the modal
confirm callback before dispatch. The remaining frontend entrypoint was
`requestTaskApply`: if it were invoked directly by a future refactor, it could
still open the Supervised Apply modal before the modal-level guard was reached.

## Options Considered

1. Keep only hidden buttons and disabled modal controls.
2. Guard `requestTaskApply` before it opens Supervised Apply.
3. Add backend mutation checks in this checkpoint.

## Decision

Guard `requestTaskApply` with the current integration bundle blocked state.
When blocked, the helper selects the task, records a local Apply blocked focus
message, emits a red toast and returns before opening Supervised Apply.

This does not open Supervised Apply, queue proposals, dispatch supervised
apply, contact backend/WING, grant live authority or close readiness.

## Why This Won

The helper is the last local frontend entrypoint before opening the modal. A
guard here makes the blocked state resilient to future UI refactors while
keeping the diff small and keeping backend safety gates unchanged.

## Rejected Alternatives

Keeping only hidden buttons was weaker because helper calls could reappear
through future action wiring. Backend mutation changes were rejected because
this checkpoint is frontend-local hardening and cannot claim backend or
hardware proof.

## Implementation Plan

1. Pass `integrationBundleApplyBlocked` into `requestTaskApply`.
2. Return before `actions.openSupervisedProposal` when the blocker is visible.
3. Add frontend contract and WING rail coverage.
4. Sync roadmap, completion-gate, capability, current-state and autopilot
   evidence docs.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m039_task_apply_request_helper_is_guarded_by_blocker -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk is limited to frontend Task Queue apply-request behavior. If it regresses,
revert the helper guard and M039 evidence/test/docs changes. The rollback does
not alter backend gates, WING transport or accepted hardware evidence.
