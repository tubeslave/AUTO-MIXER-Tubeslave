# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-sequence-plan-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify batch-preflight evidence and
emit a generic sequence-plan request template. After the generic sequence-plan
handler runs, the result still needs to be verified against the sequence-policy
provenance before dry-run.

## Options considered

1. Pass generic sequence-plan results directly to dry-run.
2. Add a read-only sequence-plan result consumer.
3. Automatically run dry-run after sequence planning.

## Decision

Use option 2. Add a read-only consumer that checks sequence-plan result order,
step safety, idempotency keys, fader/gain constraints, and no-write flags before
emitting a dry-run request template.

## Why this won

It keeps provenance explicit from trusted sequence-policy memory through
preflight and sequence planning, while preserving the existing generic dry-run
gate as a separate step.

## Rejected alternatives

Direct dry-run was rejected because it drops sequence-policy provenance checks.
Automatic dry-run was rejected because it would make the next gate implicit.

## Implementation plan

- Add the sequence-plan result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_sequence_plan_result_consumer_feeds_dry_run_without_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: the dry-run request template may be mistaken for execution authority.
Mitigation: dry-run invocation, dry-run creation, gate/executor execution,
idempotency reservation, queue mutation, live authority, and full autonomous
readiness remain false.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous preflight-result consumer remains usable.
