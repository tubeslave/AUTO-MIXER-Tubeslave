# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-execution-gate-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify dry-run results and emit a
generic execution-gate request template. After generic execution gate, the
result must still be checked against sequence-policy provenance before the
sequence executor is exposed.

## Options considered

1. Pass generic execution-gate rows directly to the sequence executor.
2. Add a read-only execution-gate result consumer.
3. Automatically run the sequence executor after gate readiness.

## Decision

Use option 2. Add a read-only consumer that verifies execution-gate result
order, ready-for-executor state, safety flags, idempotency keys, queue freshness,
and fader/gain constraints before emitting a sequence-executor request template.

## Why this won

It preserves provenance from trusted sequence-policy memory through execution
gate without invoking the live-write executor or widening live authority.

## Rejected alternatives

Direct executor use was rejected because it drops sequence-policy provenance
checks. Automatic executor invocation was rejected because execute, operator
acknowledgement, and approval must remain explicit at the live-write boundary.

## Implementation plan

- Add the execution-gate result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_dry_run_result_consumer_feeds_execution_gate_without_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: the sequence-executor request template may be mistaken for live-write
authority. Mitigation: the payload keeps executor invocation, apply attempts,
idempotency reservation, queue mutation, live authority, and full autonomous
readiness false, and defaults `execute=false`, `operator_ack=false`, and
`approved=false`.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous dry-run result consumer remains usable.
