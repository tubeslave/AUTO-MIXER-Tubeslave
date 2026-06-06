# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-20260606

## Context

Sequence-policy promotion dry-run records can describe deterministic updates
to `operator_sequence_policy_trust_gate`. The system needs an applying
boundary that remains operator-gated and isolated from live console control.

## Options considered

1. Keep the dry-run as the final layer.
2. Let accepted dry-run records mutate queue or proposal state directly.
3. Add an operator-gated executor that mutates only a supplied policy store.

## Decision

Use option 3. Add an executor that validates dry-run records and applies them
only to an explicit caller-provided `MutableMapping` when both `apply=true` and
`operator_ack=true`.

## Why this won

It advances the memory-to-policy learning chain while preserving the live
sound safety boundary. The executor creates real, testable policy-store
mutation semantics without touching live mixer transport, queues, proposal
state, models, or append-only memory.

## Rejected alternatives

- Dry-run only was rejected because the project needs a verified path toward
  policy learning.
- Direct queue/proposal mutation was rejected because policy promotion is not
  sequence execution authority.

## Implementation plan

1. Add a standalone sequence-policy promotion executor.
2. Validate dry-run envelope, status, summary counts, target policy paths,
   duplicate policy keys, duplicate idempotency keys, and unsafe authority
   claims.
3. Require `apply=true`, `operator_ack=true`, and an explicit target store
   before mutation.
4. Register the method as `shadow_only` memory-learning evidence.
5. Expose it in main pipeline preview with `apply=false`.
6. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: executor apply is mistaken for live sequence execution permission.
Mitigation: executor results keep live writes, queue mutation, model mutation,
proposal-state mutation, direct sequence execution, live authority, and full
autonomous readiness false.

Rollback: remove the executor module, registry/profile/matrix references,
preview payload, tests, ADR, and project-memory lesson.
