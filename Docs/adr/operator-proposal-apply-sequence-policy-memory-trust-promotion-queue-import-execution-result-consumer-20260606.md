# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-execution-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify execution-gate results and
emit a generic sequence-executor request template. After generic executor
results return, the result must still be checked against sequence-policy
provenance before execution audit.

## Options considered

1. Pass generic executor results directly to execution audit.
2. Add a read-only execution-result consumer.
3. Automatically run the execution audit after executor results.

## Decision

Use option 2. Add a read-only consumer that verifies executor result order,
applied status, delegated apply path, safety flags, idempotency keys, verified
readback/proof, rollback availability, and fader/gain constraints before
emitting an execution-audit request template.

## Why this won

It preserves provenance from trusted sequence-policy memory through executor
results without invoking audit or granting live authority.

## Rejected alternatives

Direct audit use was rejected because it drops sequence-policy provenance
checks. Automatic audit invocation was rejected because audit should remain a
separate explicit evidence step.

## Implementation plan

- Add the execution-result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: executor-result evidence may be mistaken for consumer live authority.
Mitigation: the consumer payload keeps audit invocation, apply attempts,
idempotency reservation, queue mutation, live writes, live authority, and full
autonomous readiness false.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous execution-gate result consumer remains usable.
