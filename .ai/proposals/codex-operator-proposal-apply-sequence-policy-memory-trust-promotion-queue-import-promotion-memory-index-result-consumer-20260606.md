# Proposal: Promotion Memory Index Result Consumer

## Short thesis

Bridge promotion memory-index evidence to trust-review request templates while
preserving promotion writer-apply provenance and fresh-context checks.

## What I understood

The generic queue-import memory path can already validate an append-only
sequence memory index before trust review. The promotion chain needs a wrapper
so indexed records remain tied to the promotion writer-apply result rather than
becoming anonymous generic evidence.

## Proposed solution

Add a promotion memory-index result consumer that validates the prior promotion
writer-apply consumer, delegates index validation to the generic consumer, and
emits a promotion-scoped memory-trust-review request template only when the
index and fresh review context pass.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/`

## Alternatives considered

Reuse the generic memory-index result consumer directly. Rejected because the
promotion chain would lose explicit linkage to the promotion writer-apply
consumer and its source dry-run provenance.

## Risks

The risk is treating indexed memory as execution authority. The consumer must
keep trust-review invocation false and keep indexed records as operator-review
evidence only.

## Test plan

Run py_compile, focused pytest for the new and adjacent consumers,
ProductState coverage, memory lint, full pytest, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer jumping directly into trust-review result validation. The safer
step is to land the provenance wrapper first so trust-review inputs are
explicitly verifiable.
