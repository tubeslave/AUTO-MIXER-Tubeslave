# ADR: Supervised Apply Integration Bundle Blocker Enforcement

## Context

M036 made the Supervised Apply modal show blocked offline integration bundle
promotion planning. A notice alone can still leave a mixed signal if task-row
Apply actions or modal confirm controls remain available while the same UI says
supervised apply remains blocked.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/local gating only and does not add backend, WING or live
transport behavior.

## Options Considered

1. Keep the M036 notice-only behavior.
2. Use the visible integration-bundle blocker as a frontend apply-control gate,
   while preserving existing backend and mode checks.
3. Add a backend queue or apply policy mutation for integration-bundle blockers.

## Decision

Use a local `integrationBundleApplyBlocked` flag derived from
`directorBlockedReasonMessages(runtime.blockedReasons)` to remove task-row
Apply actions and disable the Supervised Apply modal confirm button when
promotion planning is blocked.

This does not queue proposals, contact backend/WING, send transport, grant
readiness closure, replace backend safety gates or grant live authority.

## Why This Won

The operator-facing control state should match the blocker text. Frontend
enforcement closes the visible contradiction while keeping authoritative
console safety in existing backend, hardware and proof gates.

## Rejected Alternatives

Keeping notice-only behavior was rejected because it can leave an apply control
available next to text that says apply remains blocked.

Backend mutation was rejected because M037 is only a frontend consistency and
safety-hardening checkpoint. Queue mutation and backend apply policy changes
would need their own milestone and proof.

## Implementation Plan

- Derive `integrationBundleApplyBlocked` from Director runtime blocked-reason
  messages in `TaskQueueWorkspace`.
- Hide row-level Apply actions while the blocker is visible.
- Include the same flag in `SupervisedApplyModal` and require it to be false
  before `canRequestBackendGate` can become true.
- Disable the modal confirm button when `canRequestBackendGate` is false.
- Extend
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs` to assert
  the active source enforces the blocked state.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m037_supervised_apply_blocker_enforcement_blocks_apply_confirm -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: frontend enforcement could be mistaken for backend readiness policy. The
docs and state keep this as local UI gating only; backend and hardware gates
remain authoritative.

Rollback: revert the M037 Supervised Apply enforcement/test/docs/state changes.
The M026-M036 readiness, Director, Agent Suggestions, Task Queue and Supervised
Apply notice surfaces remain available.
