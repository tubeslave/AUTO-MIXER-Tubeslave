# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-execution-audit-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify executor results and emit a
generic execution-audit request template. After generic execution audit returns,
the result must still be checked against sequence-policy provenance before the
audit-policy gate creates policy evidence.

## Options considered

1. Pass generic execution-audit rows directly to the audit-policy gate.
2. Add a read-only execution-audit result consumer.
3. Automatically run the audit-policy gate after execution audit.

## Decision

Use option 2. Add a read-only consumer that verifies audit result order, passed
status, safety flags, idempotency keys, verified readback/proof, rollback
availability, and fader/gain constraints before emitting an audit-policy-gate
request template.

## Why this won

It preserves provenance from trusted sequence-policy memory through execution
audit without invoking the policy gate or granting policy/live authority.

## Rejected alternatives

Direct audit-policy gate use was rejected because it drops sequence-policy
provenance checks. Automatic policy-gate invocation was rejected because the
policy gate should remain a separate explicit evidence step.

## Implementation plan

- Add the execution-audit result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: audit-policy-gate request evidence may be mistaken for policy mutation or
live authority. Mitigation: the consumer payload keeps policy gate invocation,
policy mutation, runtime-memory mutation, model mutation, queue mutation, live
writes, live authority, and full autonomous readiness false.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous execution-result consumer remains usable.
