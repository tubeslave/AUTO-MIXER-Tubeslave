# Brief: Queue Import Memory Writer Apply Result Consumer

## Task

Extend the supervised sequence-policy queue-import memory chain with a read-only
consumer that verifies an explicitly applied generic append-only memory-writer
result before exposing a memory-index request template.

## Context

The queue-import chain can now verify memory-writer dry-run results and expose a
writer apply request template. The next safe step is not policy promotion or live
authority. It is to verify that the generic writer applied append-only memory
with explicit `apply=True` and `operator_ack=True`, then expose only a read-only
index request for `.ai/memory/project.md`.

## Safety Boundary

- The consumer must not invoke the writer or index handler.
- The consumer must not write memory, mutate policy/runtime/model state, reserve
  idempotency, mutate queues, or touch mixer transport.
- Source memory-file mutation may be accepted only from the generic writer
  result with `append_only_memory_write_applied`.
- The emitted memory-index request remains a template and grants no execution
  authority.

## Expected Files

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-writer-apply-result-consumer-20260606.md`

## Test Plan

- Compile touched backend modules.
- Run focused tests for the new consumer, Product Layer chain, and capability
  matrix.
- Run neighboring queue-import memory chain tests.
- Run `tools/memory/memory_lint.py`.
- Run `git diff --check`.
- Run the full pytest suite before merging.
