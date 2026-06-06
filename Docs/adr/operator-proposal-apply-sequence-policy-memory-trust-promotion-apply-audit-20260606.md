# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-apply-audit-20260606

## Context

The sequence-policy memory trust promotion executor can apply trust-state
changes to a caller-provided policy store. Applied results need read-only audit
evidence before any downstream policy consumer can use them.

## Options considered

1. Let executor apply results feed policy consumers directly.
2. Fold audit logic into the executor.
3. Add a standalone read-only apply audit.

## Decision

Use option 3. Add a transport-free apply audit that validates executor output
and emits rollback evidence records only.

## Why this won

It separates mutation from verification. The executor remains the only narrow
policy-store mutation surface, while the audit proves what changed and keeps
sequence execution, queue mutation, proposal mutation, and live console control
blocked.

## Rejected alternatives

- Direct policy consumption was rejected because rollback evidence is missing.
- Merging audit into the executor was rejected because it weakens the trust
  boundary and makes forged output harder to detect independently.

## Implementation plan

1. Add a standalone apply-audit builder.
2. Validate executor envelope, status, summary counts, apply-result target
   paths, duplicate policy keys, duplicate idempotency keys, and unsafe
   authority claims.
3. Emit rollback target policy state records with `rollback_verified=false`.
4. Register the method as `shadow_only` memory-learning evidence.
5. Expose it in main pipeline preview.
6. Record the durable safety lesson in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: audited apply results are mistaken for sequence execution permission.
Mitigation: audit records keep sequence execution, live write, queue mutation,
proposal mutation, runtime-memory mutation, and full autonomous readiness false.

Rollback: remove the audit module, registry/profile/matrix references, preview
payload, tests, ADR, and project-memory lesson.
