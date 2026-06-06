# Brief: Queue Import Promotion Acceptance Ledger Result Consumer

## Task

Extend the supervised sequence-policy queue-import memory chain with a read-only
consumer that verifies promotion acceptance-ledger results before exposing a
promotion execution-plan request template.

## Context

The chain can verify a ready promotion contract against queue-import provenance
and expose a promotion acceptance-ledger request. The next safe step is to
verify explicit operator accept/reject records against that contract provenance
and prepare the existing read-only execution-plan builder. The plan is still not
an executor and must not promote policy.

## Safety Boundary

- The consumer must not create execution plans.
- The consumer must not execute promotion candidates.
- The consumer must require at least one accepted promotion record before
  exposing an execution-plan request.
- The consumer must not mutate memory, policy, runtime state, models, queues,
  idempotency, proposal status, or live mixer transport.
- Execution planning must remain a separate read-only handler call.

## Expected Files

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-acceptance-ledger-result-consumer-20260606.md`

## Test Plan

- Compile touched backend modules.
- Run focused tests for the new consumer, Product Layer chain, and capability
  matrix.
- Run neighboring sequence-policy memory promotion tests.
- Run `tools/memory/memory_lint.py`.
- Run `git diff --check`.
