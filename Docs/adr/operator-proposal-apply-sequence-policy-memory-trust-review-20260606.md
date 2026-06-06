# ADR: operator-proposal-apply-sequence-policy-memory-trust-review-20260606

## Context

Append-only supervised sequence memory can now be indexed as structured
evidence. The project goal requires learning from prior decisions, but indexed
memory must not become policy or live execution authority without operator
review and later promotion gates.

## Options considered

1. Let indexed sequence memory influence policy immediately.
2. Leave indexed memory as passive evidence only.
3. Add a read-only operator trust-review layer.

## Decision

Use option 3. Add a trust-review builder that surfaces indexed sequence memory
as operator-review candidates only with fresh read-only sequence review context.

## Why this won

It moves the long-term learning loop forward while preserving the supervised
fader/gain boundary. Memory can become reviewable evidence, but not queue,
policy, model, runtime-memory, or live-console authority.

## Rejected alternatives

- Immediate policy influence was rejected because append-only memory and an
  index are insufficient trust gates.
- Passive-only evidence was rejected because the next policy-learning step needs
  a deterministic review-candidate contract.

## Implementation plan

1. Add a transport-free trust-review module.
2. Validate index envelope, status, summary counts, record safety flags, and
   fresh read-only review context.
3. Register as `shadow_only` memory-learning evidence.
4. Expose Product Layer preview summary, adapter, and payload.
5. Record the durable safety boundary in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_review.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py -q -k sequence_policy_memory_index`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_review.py backend/processing_registry.py backend/project_goal_capability_matrix.py backend/processing_pipeline_preview.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: operator-review candidates could be mistaken for execution authority.
Mitigation: candidate, summary, and safety flags keep execution, queue, policy,
model, runtime-memory, live authority, and full autonomous readiness false.

Rollback: remove the trust-review module, registry/profile/matrix references,
preview payload, tests, ADR, and project-memory lesson.
