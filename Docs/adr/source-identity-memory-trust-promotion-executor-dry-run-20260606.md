# ADR: source-identity-memory-trust-promotion-executor-dry-run-20260606

## Context

Source identity memory-trust promotion execution plans can surface accepted
promotion records as non-executing candidates. Before an executor can safely
apply those candidates, the project needs a typed dry-run artifact that spells
out the exact target and idempotent operation.

## Options Considered

1. Apply accepted execution candidates directly.
2. Keep execution candidates without an apply spec.
3. Add a read-only executor dry-run.

## Decision

Use option 3. Add a transport-free dry-run executor contract that creates
deterministic apply specifications for `source_identity_downstream_gate`.

## Why This Won

The dry run moves the project closer to real memory promotion by making the
future operation concrete while still preserving the current safety boundary.
It avoids hidden queue, model, runtime-memory, downstream-priority, or live
console mutation.

## Rejected Alternatives

- Direct application was rejected because rollback application and executor
  authority are not proven.
- Execution-plan-only evidence was rejected because the target operation
  remains implicit.
- Queueable dry-run records were rejected because this is not a live mixer
  action and should not enter the Product Layer task queue.

## Implementation Plan

1. Add a backend executor dry-run module.
2. Validate execution plan envelope, status, summary counts, and candidates.
3. Reject unsafe authority flags, duplicate source specs, and forged summary
   counts.
4. Emit dry-run records with target store, target path, previous/target trust
   state, rollback strategy, and idempotency key.
5. Wire dry-run status and counts into `main_pipeline_preview`.
6. Register the dry run as `shadow_only` memory-learning evidence.
7. Add focused dry-run, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_promotion_executor_dry_run.py tests/test_source_identity_calibration_memory_trust_promotion_execution_plan.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py backend/source_identity_calibration_memory_trust_promotion_execution_plan.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: dry-run apply specs could be mistaken for actual trust promotion.
Mitigation: dry-run records force `execution_allowed_by_dry_run=false`,
`memory_trust_promotion_applied=false`,
`memory_trust_promotion_allowed_by_dry_run=false`,
`downstream_priority_change_allowed_by_dry_run=false`,
`queue_mutation_allowed_by_dry_run=false`,
`model_mutation_allowed_by_dry_run=false`,
`runtime_memory_mutation_allowed_by_dry_run=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback: remove the dry-run call from `processing_pipeline_preview.py`, remove
registry/capability rows, and keep execution candidates as review evidence
only.
