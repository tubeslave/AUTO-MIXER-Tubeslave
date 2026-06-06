# Brief: Queue Import Memory Index Result Consumer

## Task

Extend the supervised sequence-policy queue-import memory chain with a read-only
consumer that verifies a sequence memory-index result before exposing a
memory-trust-review request template.

## Context

The chain can verify a writer apply result and expose an index request template.
The next step toward durable learning is to verify the index result itself and
require fresh review context before operator trust-review candidates are built.

## Safety Boundary

- The consumer must not invoke the trust-review handler.
- The consumer must not mutate memory, policy, runtime state, models, queues,
  idempotency, proposal status, or live mixer transport.
- The consumer must require a read-only memory index and fresh read-only
  sequence-review context.
- Indexed memory remains operator-review evidence only.

## Expected Files

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-index-result-consumer-20260606.md`

## Test Plan

- Compile touched backend modules.
- Run focused tests for the new consumer, Product Layer chain, and capability
  matrix.
- Run neighboring queue-import memory-chain tests.
- Run `tools/memory/memory_lint.py`.
- Run `git diff --check`.
- Run full pytest before merge.
