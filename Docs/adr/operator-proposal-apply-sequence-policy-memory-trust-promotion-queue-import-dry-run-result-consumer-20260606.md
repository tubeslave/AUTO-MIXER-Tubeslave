# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-dry-run-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify sequence planning and emit a
generic dry-run request template. After generic dry-run, the result must still
be checked against sequence-policy provenance before execution gating.

## Options considered

1. Pass generic dry-run results directly to the execution gate.
2. Add a read-only dry-run result consumer.
3. Automatically run the execution gate after dry-run.

## Decision

Use option 2. Add a read-only consumer that verifies dry-run result order,
would-apply state, safety flags, idempotency keys, and fader/gain constraints
before emitting an execution-gate request template.

## Why this won

It preserves provenance from trusted sequence-policy memory through dry-run
without invoking the next gate or widening live authority.

## Rejected alternatives

Direct execution-gate use was rejected because it drops sequence-policy
provenance checks. Automatic execution-gate invocation was rejected because
operator acknowledgement must remain explicit.

## Implementation plan

- Add the dry-run result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_dry_run_result_consumer_feeds_execution_gate_without_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: the execution-gate request template may be mistaken for execution or live
authority. Mitigation: the payload keeps gate invocation, executor execution,
idempotency reservation, queue mutation, live authority, and full autonomous
readiness false, and defaults `operator_ack=false`.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous sequence-plan result consumer remains usable.
