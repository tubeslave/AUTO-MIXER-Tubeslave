# Brief: Queue Import Memory Trust Acceptance Ledger Result Consumer

## Task

Extend the supervised sequence-policy queue-import memory chain with a read-only
consumer that verifies operator-supplied memory trust acceptance ledgers before
exposing a promotion-plan request template.

## Context

The chain can now verify memory trust-review candidates and emit acceptance
ledger templates with empty `acceptance_records`. The next safe step is not to
auto-accept those candidates. It is to verify a later operator-supplied
accepted/rejected ledger and prepare the existing read-only promotion-plan
handler request.

## Safety Boundary

- The consumer must not create accepted/rejected records.
- The consumer must block empty or unreviewed acceptance ledgers.
- The emitted request must be a read-only promotion-plan template only.
- The consumer must not mutate memory, policy, runtime state, models, queues,
  idempotency, proposal status, or live mixer transport.
- Promotion-plan creation must remain a separate handler call.

## Expected Files

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-trust-acceptance-ledger-result-consumer-20260606.md`

## Test Plan

- Compile touched backend modules.
- Run focused tests for the new consumer, Product Layer chain, and capability
  matrix.
- Run neighboring queue-import memory-chain tests.
- Run `tools/memory/memory_lint.py`.
- Run `git diff --check`.
- Run full pytest before merge.
