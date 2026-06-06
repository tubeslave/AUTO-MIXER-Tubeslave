# Brief: Queue Import Promotion Plan Result Consumer

## Task

Extend the supervised sequence-policy queue-import memory chain with a read-only
consumer that verifies promotion-plan results before exposing a promotion
contract request template.

## Context

The chain can verify operator accepted/rejected memory trust ledgers and request
the existing promotion-plan builder. The next safe step is to verify that the
promotion plan matches that queue-import acceptance provenance and prepare the
existing promotion-contract handler request. The contract still requires fresh
sequence-policy context and rollback evidence.

## Safety Boundary

- The consumer must not create promotion contracts.
- The consumer must not promote policy or execute proposals.
- The emitted request must require fresh sequence-policy context and rollback
  evidence.
- The consumer must not mutate memory, policy, runtime state, models, queues,
  idempotency, proposal status, or live mixer transport.
- Contract creation must remain a separate read-only handler call.

## Expected Files

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-plan-result-consumer-20260606.md`

## Test Plan

- Compile touched backend modules.
- Run focused tests for the new consumer, Product Layer chain, and capability
  matrix.
- Run neighboring sequence-policy memory promotion tests.
- Run `tools/memory/memory_lint.py`.
- Run `git diff --check`.
- Run full pytest before merge.
