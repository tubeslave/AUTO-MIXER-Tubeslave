# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-plan-20260606

## Context

Indexed supervised sequence memory can become trust-review candidates and then
operator accepted/rejected ledger records. The next step is to show which
accepted records are candidates for manual promotion without changing policy or
execution behavior.

## Options considered

1. Treat accepted ledger records as immediately policy-trusted.
2. Skip straight to a sequence-policy executor.
3. Add a read-only manual promotion plan.

## Decision

Use option 3. Add a transport-free promotion plan that validates the acceptance
ledger and surfaces accepted records only as manual promotion candidates.

## Why this won

It moves long-term learning toward autonomy while preserving the current
supervised fader/gain live boundary. Accepted records become reviewable evidence
for a future contract, not queue imports, policy updates, runtime memory writes,
or live console actions.

## Rejected alternatives

- Direct policy trust was rejected because accepted evidence still needs a
  promotion contract and rollback/audit boundaries.
- Direct executor work was rejected because sequence memory must not bypass
  `apply_operator_proposal` or the existing supervised apply gates.

## Implementation plan

1. Add a standalone promotion-plan builder.
2. Validate ledger envelope, status, summary counts, accepted record safety,
   duplicate reviews, UTC timestamps, and unsafe authority claims.
3. Register the method as `shadow_only` memory-learning evidence.
4. Expose it in main pipeline preview.
5. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: promotion candidates are mistaken for execution authority. Mitigation:
plan candidates and summary keep promotion, execution, queue, policy, model,
runtime-memory, live authority, and full autonomous readiness false.

Rollback: remove the promotion-plan module, registry/profile/matrix references,
preview payload, tests, ADR, and project-memory lesson.
