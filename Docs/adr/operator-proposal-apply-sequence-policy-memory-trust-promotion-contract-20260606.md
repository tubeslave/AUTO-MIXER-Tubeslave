# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-contract-20260606

## Context

Accepted sequence-memory trust records can become manual promotion candidates.
Before any later policy or execution influence is considered, the system needs
fresh sequence-policy context and rollback evidence.

## Options considered

1. Let promotion-plan candidates directly influence sequence policy.
2. Combine promotion contract and executor.
3. Add a read-only promotion contract.

## Decision

Use option 3. Add a transport-free contract that validates promotion-plan
candidates, fresh sequence-policy context, and rollback evidence, then surfaces
contract records only for operator review.

## Why this won

It moves long-term learning toward autonomous policy improvement while
preserving the current supervised fader/gain live boundary. Contract records are
evidence for a future executor, not policy updates, queue imports, runtime memory
writes, or live console actions.

## Rejected alternatives

- Direct policy influence was rejected because accepted memory still needs
  rollback-aware review.
- A combined executor was rejected because contract review must remain separate
  from policy mutation and live execution.

## Implementation plan

1. Add a standalone contract builder.
2. Validate promotion-plan envelope, summary counts, candidates, fresh context,
   rollback evidence, UTC timestamps, and unsafe authority claims.
3. Register the method as `shadow_only` memory-learning evidence.
4. Expose it in main pipeline preview.
5. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: contract records are mistaken for execution authority. Mitigation:
contract records and summary keep promotion, execution, queue, policy, model,
runtime-memory, live authority, and full autonomous readiness false.

Rollback: remove the contract module, registry/profile/matrix references,
preview payload, tests, ADR, and project-memory lesson.
