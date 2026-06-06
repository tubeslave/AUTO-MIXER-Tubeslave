# Proposal: batch-preflight result consumer for promotion queue import

## Short thesis

Add a read-only result consumer between the promotion batch-preflight request and the generic sequence-plan handler. It should verify promotion provenance and approved batch-preflight readiness, then emit only a sequence-plan request template.

## What I understood

The current promotion chain can verify applied queue import, queue-import preflight planning, and expose a batch-preflight request. Batch preflight is still not sequence-plan authority unless its output matches the promotion plan and all selected rows are ready.

## Proposed solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer`.
- Verify the prior promotion preflight-plan result consumer envelope, safety flags, candidate ids, and emitted batch-preflight request.
- Reuse the existing generic queue-import preflight-result consumer to validate approved batch-preflight output against the underlying queue-import preflight plan.
- Reject forged evidence such as wrong order, missing approval, live-write flags, reserved idempotency, or blocked preflight rows.
- Emit only a `plan_operator_proposal_apply_sequence` request template with preflight results when all evidence is ready.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Let the preflight-plan result consumer emit a sequence-plan request directly. Rejected because batch preflight readiness must be verified first.
- Duplicate all generic batch-preflight validation logic. Rejected because the existing queue-import preflight-result consumer already owns that contract.
- Invoke the sequence-plan handler from inside this consumer. Rejected because result consumers should expose request templates, not execute the next Product Layer step.

## Risks

- The wrapper can hide generic validation details. Mitigation: invalid generic evidence is surfaced under `generic_preflight_result_consumer:*` invalid ids and focused tests cover forged paths.
- The request can look close to execution authority. Mitigation: it remains sequence-plan-only; the sequence plan, dry-run, gate, executor, audit, and supervised apply gates still remain separate.

## Test plan

- Focused pytest for the new consumer.
- Product Layer dispatch pytest through generic sequence planning.
- Registry and capability matrix checks.
- `py_compile`, `memory_lint`, and `git diff --check`.

## Where the other agent may disagree

Kimi might prefer to inline the generic validation to make this file self-contained. I would reuse the existing validator because it keeps one authoritative definition of batch-preflight readiness.
