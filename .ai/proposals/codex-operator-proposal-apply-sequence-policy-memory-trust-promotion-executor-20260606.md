# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-20260606

## Short thesis

Sequence-policy promotion can move from dry-run evidence to a narrow,
operator-gated executor that mutates only an explicit policy store.

## What I understood

The project needs real learning progress toward supervised automation, but
policy promotion cannot be confused with live execution. The dry-run already
defines deterministic target paths and idempotency keys. The executor should
validate those specs again and require explicit acknowledgement and a supplied
store before applying them.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py`.
It consumes the dry-run payload, validates envelope, summary, and record-level
safety, then optionally writes target trust states into a supplied
`MutableMapping`. Product Layer preview calls it with `apply=false`, so preview
is non-mutating.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-executor-20260606.md`

## Alternatives considered

- Keep only the dry-run. Rejected: the project goal needs a verified path from
  accepted learning evidence to policy-store updates.
- Let the executor mutate the real queue or proposal state. Rejected: policy
  learning is not proposal execution authority.

## Risks

- A forged dry-run could claim policy or runtime-memory authority. The executor
  rejects unsafe envelope, summary, and record-level claims.
- Applying to the wrong object could affect runtime behavior. The executor
  requires a caller-provided target store and Product Layer preview passes none.

## Test plan

- Focused executor tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may argue that any mutation surface should wait for rollback audit. This
executor is deliberately scoped to an explicit in-memory store and will be
followed by an apply audit before any wider policy consumer can use it.
