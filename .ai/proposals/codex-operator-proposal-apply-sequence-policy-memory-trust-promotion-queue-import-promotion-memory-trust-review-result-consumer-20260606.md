# Proposal: Promotion Memory Trust Review Result Consumer

## Short thesis

Bridge promotion memory trust-review evidence to acceptance-ledger request
templates while preserving promotion memory-index provenance and operator
decision boundaries.

## What I understood

The generic queue-import memory path can validate trust-review candidates before
acceptance. The promotion chain needs a wrapper so those candidates remain tied
to the promotion memory-index result rather than becoming anonymous generic
evidence.

## Proposed solution

Add a promotion memory-trust-review result consumer that validates the prior
promotion memory-index consumer, delegates trust-review result validation to the
generic consumer, and emits a promotion-scoped acceptance-ledger request
template only when candidate evidence matches.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/`

## Alternatives considered

Reuse the generic memory trust-review result consumer directly. Rejected
because the promotion chain would lose explicit linkage to promotion
memory-index provenance and fresh sequence-review context.

## Risks

The risk is treating trust-review candidates as operator acceptance. The
consumer must leave acceptance records empty, require explicit accept/reject
decisions, and keep all policy, queue, proposal, trust-store, and live authority
flags false.

## Test plan

Run py_compile, focused pytest for the new and adjacent consumers,
ProductState coverage, memory lint, full pytest, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer jumping directly into acceptance-ledger result validation. The
safer step is to land the provenance wrapper first so acceptance inputs are
explicitly verifiable.
