# Proposal: Promotion Memory Trust Acceptance Ledger Result Consumer

## Short thesis

Bridge promotion memory trust acceptance-ledger evidence to promotion-plan
request templates while preserving promotion trust-review provenance and
operator decision boundaries.

## What I understood

The generic queue-import memory path can validate acceptance ledgers before
promotion planning. The promotion chain needs a wrapper so explicit
accept/reject records remain tied to the promotion memory-trust-review result
rather than becoming anonymous generic evidence.

## Proposed solution

Add a promotion memory-trust-acceptance-ledger result consumer that validates
the prior promotion trust-review consumer, delegates acceptance-ledger
validation to the generic consumer, and emits a promotion-scoped promotion-plan
request template only when operator decisions match.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer.py`
- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/`

## Alternatives considered

Reuse the generic acceptance-ledger result consumer directly. Rejected because
the promotion chain would lose explicit linkage to promotion memory-trust-review
provenance and its source memory-index evidence.

## Risks

The risk is treating operator acceptance as policy promotion. The consumer must
only emit a promotion-plan request template and keep promotion planning,
policy-store mutation, proposal mutation, sequence-policy trust-store mutation,
and live authority false.

## Test plan

Run py_compile, focused pytest for the new and adjacent consumers,
ProductState coverage, memory lint, full pytest, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer using only the generic acceptance consumer. The safer promotion
chain keeps provenance explicit before promotion planning.
