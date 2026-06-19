# ADR: Operator Adapter Integration Bundle Readiness Summary

## Context

M026 added an `offline_integration_bundle` overlay to autonomy promotion
readiness. That backend snapshot now blocks promotion planning when offline
integration bundle evidence is missing or contract-blocked.

The operator-facing module status adapter already renders autonomy promotion
rows and a summary row, but it did not expose the new integration bundle
readiness status in that summary.

M001 still waits for a real WING read-only artifact and operator review. This
decision is frontend/read-only only.

## Options Considered

1. Leave the integration bundle blocker visible only in raw backend snapshots.
2. Add integration bundle tags to the existing autonomy readiness summary row.
3. Add a separate operator action row that looks like an actionable task.

## Decision

Add read-only integration bundle fields to the existing
`autonomy_promotion_summary` module status row in
`frontend/src/services/operatorAdapters.ts`.

The adapter surfaces readiness, status, source authority denial, blockers,
missing checks and child contract statuses as module status tags and text.
It keeps `canWriteLive: false` and `liveWriteKinds: []`.

## Why This Won

The existing summary row is already the operator-facing place for aggregate
autonomy readiness. Extending it keeps the M026 blocker visible without adding
a new action surface or implying that frontend state can approve promotion.

## Rejected Alternatives

Leaving the blocker only in raw backend JSON was rejected because operators and
reviewers need the same fail-closed integration status in module status rows.

A separate actionable task row was rejected because the integration bundle
state is evidence/status, not an apply command.

## Implementation Plan

- Read `summary.offline_integration_bundle` in
  `frontend/src/services/operatorAdapters.ts`.
- Add `integration_bundle_*` tags to `autonomy_promotion_summary`.
- Include integration bundle blockers in `blockedReason` and `nextStep`.
- Cover the behavior in
  `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`.
- Update roadmap, completion gate, capability matrix, current state and
  autopilot evidence.

## Test Plan

- `npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py::test_m027_operator_adapter_surfaces_integration_bundle_readiness_summary -q`
- `PYTHONPATH=backend python -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

Risk: the summary row becomes noisier when an integration bundle is blocked.
This is acceptable because the blocker is promotion-critical and read-only.

Rollback: revert the M027 adapter/test/docs/state changes. The M026 backend
readiness overlay remains available for raw snapshot consumers.
