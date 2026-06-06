# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-preflight-result-consumer-20260606

## Context

Sequence-policy queue imports can produce pending operator proposals and a
read-only batch-preflight request template. Before the generic sequence-plan
handler is used, the system needs evidence that the batch-preflight payload
matches the sequence-policy import order and safety envelope.

## Options considered

1. Let operators pass batch-preflight results directly to generic sequence
   planning.
2. Add a read-only preflight-result consumer.
3. Automatically run sequence planning after batch preflight.

## Decision

Use option 2. Add a read-only preflight-result consumer that verifies the
preflight plan and approved batch-preflight result, then emits a sequence-plan
request template only when all planned candidates are ready and ordered.

## Why this won

It preserves provenance from sequence-policy memory into apply-sequence planning
without invoking handlers or widening authority.

## Rejected alternatives

Direct generic sequence planning was rejected because it lacks sequence-policy
import provenance checks. Automatic sequence planning was rejected because
operator action boundaries must stay explicit.

## Implementation plan

- Add the verifier builder.
- Add a Product Layer handler that only returns the verifier payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_preflight_result_consumer_feeds_sequence_plan_without_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: the sequence-plan request template may be mistaken for execution
authority. Mitigation: the payload keeps handler invocation, sequence-plan
creation, direct execution, idempotency reservation, queue mutation, live
authority, and full autonomous readiness false.

Rollback: remove the verifier handler/module and registry/matrix references.
The earlier queue-import preflight-plan bridge remains usable.
