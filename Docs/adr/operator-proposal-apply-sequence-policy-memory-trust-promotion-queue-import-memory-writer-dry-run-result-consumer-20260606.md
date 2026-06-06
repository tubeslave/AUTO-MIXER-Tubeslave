# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-writer-dry-run-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify memory drafts and emit a
generic memory-writer request template. The generic memory writer can mutate
`.ai/memory/project.md` when both `apply=true` and `operator_ack=true`. A dry-run
result must therefore be verified before any apply request template is exposed.

## Options considered

1. Pass memory-writer request templates directly to apply.
2. Add a read-only memory-writer dry-run result consumer.
3. Automatically apply memory writes after a successful dry-run.

## Decision

Use option 2. Add a read-only consumer that verifies upstream consumer status,
writer dry-run envelope safety, false write/mutation flags, no write results,
target path, and the expected operator-ack/apply blocker before emitting an
apply request template.

## Why this won

It proves the writer accepted the memory draft records but stopped before file
mutation, preserving sequence-policy provenance without writing memory.

## Rejected alternatives

Direct apply templates were rejected because they skip dry-run proof. Automatic
apply was rejected because memory writes must remain an explicit operator action.

## Implementation plan

- Add the memory-writer dry-run result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage through apply request
  template.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: memory-writer apply request evidence may be mistaken for memory mutation
or live authority. Mitigation: the consumer payload keeps writer invocation,
memory-file mutation, policy mutation, runtime-memory mutation, model mutation,
queue mutation, live writes, live authority, and full autonomous readiness false.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous memory-draft result consumer remains usable.
