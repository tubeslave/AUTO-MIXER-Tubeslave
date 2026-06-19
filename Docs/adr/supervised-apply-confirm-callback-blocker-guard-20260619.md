# ADR: Supervised Apply Confirm Callback Blocker Guard

## Context

M037 removed task-row Apply actions and disabled the Supervised Apply modal
confirm button when offline integration-bundle blockers were visible. Disabled
controls are necessary, but the callback itself should also fail closed so a
future refactor, keyboard path or programmatic invocation cannot dispatch
`confirmSupervisedApply` while the blocker is active.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/local callback guarding only and does not add backend,
WING or live transport behavior.

## Options Considered

1. Rely on disabled controls from M037.
2. Add a local guarded confirm callback before dispatching the existing action.
3. Add a backend queue or apply policy mutation for integration-bundle
   blockers.

## Decision

Use `guardedConfirmSupervisedApply` in `TaskQueueWorkspace`. The function
returns before calling `actions.confirmSupervisedApply` while
`integrationBundleApplyBlocked` is true.

This does not queue proposals, contact backend/WING, send transport, grant
readiness closure, replace backend safety gates or grant live authority.

## Why This Won

The dispatch path should match the disabled-control state. A local guard adds a
small fail-closed layer without changing authoritative backend or hardware
safety policy.

## Rejected Alternatives

Relying only on disabled controls was rejected because it leaves the callback
unguarded and makes future UI changes easier to get wrong.

Backend mutation was rejected because M038 is only a frontend callback safety
checkpoint. Backend apply policy changes would need their own milestone and
proof.

## Implementation Plan

- Add `guardedConfirmSupervisedApply` near the existing integration-bundle
  blocker derivation in `TaskQueueWorkspace`.
- Return before dispatch when `integrationBundleApplyBlocked` is true.
- Route modal `onConfirm` through the guarded function.
- Extend
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to assert
  the active source includes the guarded dispatch path.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m038_supervised_apply_confirm_callback_is_guarded_by_blocker -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: the local guard could be mistaken for backend readiness policy. The docs
and state keep this as frontend-only dispatch hardening; backend and hardware
gates remain authoritative.

Rollback: revert the M038 guarded-callback/test/docs/state changes. The
M026-M037 readiness, Director, Agent Suggestions, Task Queue, Supervised Apply
notice and Supervised Apply enforcement surfaces remain available.
