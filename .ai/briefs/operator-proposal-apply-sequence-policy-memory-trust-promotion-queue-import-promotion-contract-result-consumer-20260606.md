# Brief: Queue Import Promotion Contract Result Consumer

## Task

Extend the supervised sequence-policy queue-import memory chain with a read-only
consumer that verifies promotion-contract results before exposing a promotion
acceptance-ledger request template.

## Context

The chain can verify operator accepted/rejected memory trust ledgers, request a
promotion plan, and verify that promotion-plan candidates match queue-import
provenance before requesting the existing promotion-contract builder. The next
safe step is to verify a ready contract against that provenance and prepare the
existing promotion acceptance-ledger handler request. Acceptance still requires
explicit operator-supplied records.

## Safety Boundary

- The consumer must not accept promotion contracts.
- The consumer must leave `acceptance_records` empty by default.
- The consumer must not invoke the promotion acceptance-ledger handler.
- The consumer must not promote policy or execute proposals.
- The consumer must not mutate memory, policy, runtime state, models, queues,
  idempotency, proposal status, or live mixer transport.
- Acceptance recording must remain a separate read-only handler call.

## Expected Files

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-contract-result-consumer-20260606.md`

## Test Plan

- Compile touched backend modules.
- Run focused tests for the new consumer, Product Layer chain, and capability
  matrix.
- Run neighboring sequence-policy memory promotion tests.
- Run `tools/memory/memory_lint.py`.
- Run `git diff --check`.
- Run full pytest before merge.
