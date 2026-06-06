# ADR: operator-proposal-apply-sequence-policy-memory-trust-acceptance-ledger-20260606

## Context

Indexed supervised sequence memory can now become operator-review candidates
when fresh read-only context is present. The next learning step is to record the
operator's acceptance decision without granting policy or execution authority.

## Options considered

1. Treat every trust-review candidate as accepted.
2. Let accepted records directly influence sequence policy.
3. Add a read-only acceptance ledger.

## Decision

Use option 3. Add a transport-free ledger that validates operator accepted or
rejected decisions against current sequence-memory trust-review candidates.

## Why this won

It creates an auditable learning signal while preserving the current supervised
fader/gain live boundary. Accepted records can feed a future promotion contract,
but cannot mutate queue, policy, model, runtime memory, or live console state.

## Rejected alternatives

- Automatic acceptance was rejected because it removes operator judgement.
- Direct policy influence was rejected because acceptance is not a promotion
  gate or executor.

## Implementation plan

1. Add a standalone ledger builder.
2. Validate review envelope, summary, candidate safety, acceptance links, UTC
   timestamps, and unsafe authority claims.
3. Register the method as `shadow_only` memory-learning evidence.
4. Expose it in main pipeline preview.
5. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: accepted records are mistaken for execution authority. Mitigation: ledger
records and summary keep promotion, execution, queue, policy, model,
runtime-memory, live authority, and full autonomous readiness false.

Rollback: remove the ledger module, registry/profile/matrix references, preview
payload, tests, ADR, and project-memory lesson.
