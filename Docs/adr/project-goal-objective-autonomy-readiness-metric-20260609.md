# project-goal-objective-autonomy-readiness-metric-20260609

## Context

The objective-alignment snapshot already exposes which project-goal capabilities are
missing, blocked, proposal-only, or already ready for supervised execution. For long-term
autonomy planning and operator visibility, a single normalized score is also needed
without changing live-write behavior.

## Options considered

- Keep only raw counts (`missing_row_count`, `blocked_live_row_count`, etc.) and
  require consumers to recompute any readiness figure themselves.
- Add a separate, unstructured metric endpoint outside existing snapshots.
- Extend existing alignment snapshots and summaries with explicit readiness metrics.

## Decision

Add three fields to `build_project_goal_objective_alignment_snapshot()`:
- `autonomy_readiness_score`
- `autonomy_readiness_ready_row_count`
- `autonomy_readiness_gap_row_count`

Propagate these fields through existing read-only runtime/read/write contracts:
- `processing_registry` snapshot
- `operator_product_state` dashboard runtime
- `processing_pipeline_preview` summary
- frontend module status adapter and related contract tests.

The score is computed as:
`int((ready_row_count / required_row_count) * 100)` where ready rows include status
`canonical` and `supervised_narrow`.

## Why this won

A compact metric gives operator-facing views a single measure of autonomy progression
while preserving row-level evidence and without introducing new live execution paths.
This supports objective tracking for long-term autonomous capability growth without
changing the existing safety boundary.

## Rejected alternatives

Raw-count-only reporting was rejected because it fragments readiness logic across
consumers and weakens auditability. A separate endpoint was rejected as duplication
against the existing objective snapshots and dashboard contracts.

## Implementation plan

1. Compute the three readiness fields in `backend/project_goal_capability_matrix.py`.
2. Add pass-through fields in `backend/processing_registry.py`,
   `backend/operator_product_state.py`, and `backend/processing_pipeline_preview.py`.
3. Surface readiness fields in `frontend/src/services/operatorAdapters.ts` module
   profile tokens.
4. Extend contract tests for snapshots and adapters to assert field propagation and
   numerical consistency.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_project_goal_capability_matrix.py tests/test_processing_registry.py tests/test_operator_product_state.py -q`
- `cd frontend && npm test`

## Risks and rollback

The score definition is a reporting choice and should match product decisions. If
interpretation changes (for example, including `available` status as partially ready),
update this ADR and adjust the upstream calculation with coordinated snapshot tests.
Rollback is reverting this ADR, the matrix/projection fields, and associated tests.
