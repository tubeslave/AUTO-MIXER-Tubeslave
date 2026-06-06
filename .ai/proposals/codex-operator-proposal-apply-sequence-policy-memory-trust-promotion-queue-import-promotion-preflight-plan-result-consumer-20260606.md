# Proposal: preflight-plan result consumer for promotion queue import

## Short thesis

Add a read-only result consumer between the queue-import preflight planner and the existing batch preflight handler. It should verify that the preflight plan matches applied queue-importer provenance, then emit only a batch-preflight request template.

## What I understood

The current promotion chain can verify an applied queue-importer result and produce a preflight-plan request. That plan still is not apply authority: it only selects pending imported proposals and builds request specs for the generic batch preflight handler. The next safe step is to consume that plan as evidence before exposing the batch preflight template.

## Proposed solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer`.
- Verify the prior queue-importer result consumer envelope, safety flags, imported proposal ids, and emitted preflight-plan request.
- Verify the queue-import preflight-plan envelope, safety flags, candidate ids, request specs, and batch-preflight request template.
- Reject forged plan authority such as `approved=true`, `live_write=true`, handler invocation claims, queue mutation, or idempotency reservation.
- Emit only a `preflight_operator_proposal_apply_batch` request template with `approved=false` and explicit `operator_must_set_approved_true=true`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Let the queue-importer result consumer emit a batch-preflight request directly. Rejected because preflight candidate selection belongs to the queue-import preflight planner.
- Let this consumer run the batch preflight handler. Rejected because result consumers should verify provenance and expose request templates, not execute the next Product Layer handler.
- Mark the emitted batch request as approved. Rejected because approval must remain an explicit operator action at the batch preflight boundary.

## Risks

- The verifier may be too strict if the queue-import preflight-plan schema evolves. Mitigation: checks are scoped to current plan fields and focused tests cover valid and forged paths.
- The batch preflight request can appear close to apply authority. Mitigation: the request remains `approved=false`, batch preflight itself is read-only, and real writes still require `apply_operator_proposal`.

## Test plan

- Focused pytest for the new consumer.
- Product Layer dispatch pytest for applied import -> preflight plan -> consumer -> batch preflight.
- Registry and capability matrix checks.
- `py_compile`, `memory_lint`, `git diff --check`, and full pytest before commit.

## Where the other agent may disagree

Kimi might prefer to merge this verifier into the generic preflight-plan handler. I would keep it separate because the generic handler should stay reusable, while this consumer enforces promotion-chain provenance and template-only safety.
