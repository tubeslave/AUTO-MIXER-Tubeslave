# Brief: Queue Import Memory Trust Review Result Consumer

## Task

Extend the supervised sequence-policy queue-import memory chain with a read-only
consumer that verifies memory trust-review results before exposing an operator
acceptance-ledger request template.

## Context

The chain can verify memory index results and expose a memory trust-review
request. The next step is not automatic acceptance or policy promotion. It is to
verify the trust-review result and prepare acceptance record templates that an
operator must explicitly complete.

## Safety Boundary

- The consumer must not invoke the acceptance-ledger handler.
- The consumer must not create accepted/rejected records by default.
- The emitted request must keep `acceptance_records` empty until an operator
  supplies explicit decisions.
- The consumer must not mutate memory, policy, runtime state, models, queues,
  idempotency, proposal status, or live mixer transport.

## Expected Files

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-trust-review-result-consumer-20260606.md`

## Test Plan

- Compile touched backend modules.
- Run focused tests for the new consumer, Product Layer chain, and capability
  matrix.
- Run neighboring queue-import memory-chain tests.
- Run `tools/memory/memory_lint.py`.
- Run `git diff --check`.
- Run full pytest before merge.
