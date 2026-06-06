# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-dry-run-20260606

## Context

Sequence-memory trust promotion execution candidates identify accepted
operator-reviewed records that may later influence supervised apply-sequence
policy. Before any mutation surface exists, the system needs a deterministic
dry-run contract that shows the target policy operation and blocks apply.

## Options considered

1. Treat execution-plan candidates as directly executable.
2. Build the applying executor immediately.
3. Add a read-only executor dry-run.

## Decision

Use option 3. Add a transport-free dry-run builder that validates the execution
plan and emits deterministic specs for `operator_sequence_policy_trust_gate`.

## Why this won

It advances the autonomy chain toward policy learning while preserving the
current safety boundary. The dry-run makes future mutation explicit, testable,
and reviewable without changing queues, policy stores, runtime memory, or live
console state.

## Rejected alternatives

- Direct execution was rejected because candidates are evidence, not authority.
- Immediate executor implementation was rejected because it would mix dry-run
  proof with a new mutation surface.

## Implementation plan

1. Add a standalone executor dry-run builder.
2. Validate execution-plan envelope, status, summary counts, candidate
   identity, duplicate policy keys, UTC timestamps, and unsafe authority claims.
3. Emit deterministic target policy paths, operations, idempotency keys, and
   rollback fields.
4. Register the method as `shadow_only` memory-learning evidence.
5. Expose it in main pipeline preview.
6. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: dry-run specs are mistaken for execution permission. Mitigation: dry-run
records and summary keep execution, sequence-policy promotion, queue, policy,
model, runtime-memory, live authority, and full autonomous readiness false.

Rollback: remove the dry-run module, registry/profile/matrix references,
preview payload, tests, ADR, and project-memory lesson.
