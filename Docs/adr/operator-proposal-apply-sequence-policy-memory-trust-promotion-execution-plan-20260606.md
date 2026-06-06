# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-execution-plan-20260606

## Context

Sequence-memory trust promotion acceptance records can mark promotion contract
records as accepted or rejected by the operator. The next step is to expose
accepted records as future executor candidates without granting sequence-policy
or live execution authority.

## Options considered

1. Let accepted promotion records directly mutate sequence policy.
2. Build a full executor immediately.
3. Add a read-only execution plan for accepted records.

## Decision

Use option 3. Add a transport-free execution plan that validates the promotion
acceptance ledger and surfaces only non-executing candidates for later review.

## Why this won

It preserves the supervised fader/gain live boundary while making the next
learning handoff explicit. The plan can be reviewed and tested before any
executor or policy mutation surface exists.

## Rejected alternatives

- Direct policy mutation was rejected because accepted records are evidence,
  not authority.
- Immediate executor implementation was rejected because it would combine two
  trust-boundary changes in one step.

## Implementation plan

1. Add a standalone execution-plan builder.
2. Validate acceptance ledger envelope, summary counts, accepted records,
   ambiguous contract reviews, UTC timestamps, and unsafe authority claims.
3. Register the method as `shadow_only` memory-learning evidence.
4. Expose it in main pipeline preview.
5. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: accepted promotion candidates are mistaken for execution permission.
Mitigation: plan records and summary keep execution, sequence-policy promotion,
queue, policy, model, runtime-memory, live authority, and full autonomous
readiness false.

Rollback: remove the execution-plan module, registry/profile/matrix references,
preview payload, tests, ADR, and project-memory lesson.
