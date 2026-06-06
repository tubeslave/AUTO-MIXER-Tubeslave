# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-draft-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify audit-policy gate results and
emit a generic policy-memory draft request template. After generic memory draft
returns `memory_draft_ready` records, those records must still be checked
against sequence-policy queue-import provenance before the append-only memory
writer is exposed.

## Options considered

1. Pass generic memory draft output directly to the writer.
2. Add a read-only memory-draft result consumer.
3. Automatically run the writer dry-run after memory draft.

## Decision

Use option 2. Add a read-only consumer that verifies upstream consumer status,
memory-draft envelope safety, record order, draft IDs, idempotency keys,
append-only target, verified evidence, rollback availability, and fader/gain
constraints before emitting a memory-writer request template.

## Why this won

It preserves provenance from trusted sequence-policy memory through queue
import, execution, audit, policy gate, and memory draft without invoking the
writer or writing memory.

## Rejected alternatives

Direct writer use was rejected because it drops sequence-policy provenance
checks. Automatic writer dry-run was rejected because the writer should remain a
separate explicit evidence step before any append-only write is allowed.

## Implementation plan

- Add the memory-draft result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage through writer dry-run.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: memory-writer request evidence may be mistaken for memory mutation or live
authority. Mitigation: the consumer payload keeps writer invocation, memory-file
mutation, memory-store mutation, policy mutation, runtime-memory mutation, model
mutation, queue mutation, live writes, live authority, and full autonomous
readiness false.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous audit-policy-gate result consumer remains usable.
