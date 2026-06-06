# ADR: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-audit-policy-gate-result-consumer-20260606

## Context

The sequence-policy queue-import path can verify execution audits and emit a
generic audit-policy-gate request template. After the generic policy gate
returns `policy_evidence_ready` rows, that evidence must still be checked
against sequence-policy queue-import provenance before a policy-memory draft
request is exposed.

## Options considered

1. Pass generic audit-policy gate output directly to memory draft.
2. Add a read-only audit-policy-gate result consumer.
3. Automatically run the memory draft handler after the policy gate.

## Decision

Use option 2. Add a read-only consumer that verifies upstream consumer status,
policy-gate envelope safety, row order, idempotency keys, ready policy evidence,
verified readback/proof, rollback availability, and fader/gain constraints
before emitting a memory-draft request template.

## Why this won

It preserves provenance from trusted sequence-policy memory through queue
import, preflight, sequence planning, dry-run, execution gate, execution,
execution audit, and policy gate without invoking memory draft or granting
memory-write/live authority.

## Rejected alternatives

Direct memory-draft use was rejected because it drops sequence-policy provenance
checks. Automatic memory-draft invocation was rejected because memory learning
must remain an explicit separate evidence step before any writer exists.

## Implementation plan

- Add the audit-policy-gate result consumer builder.
- Add a Product Layer handler that only returns the consumer payload.
- Register the method as `shadow_only`.
- Track it in the project-goal capability matrix.
- Add focused unit tests and Product Layer chain coverage through memory draft.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`

## Risks and rollback

Risk: memory-draft request evidence may be mistaken for memory mutation or live
authority. Mitigation: the consumer payload keeps memory-draft invocation,
memory mutation, policy mutation, runtime-memory mutation, model mutation, queue
mutation, live writes, live authority, and full autonomous readiness false.

Rollback: remove the consumer handler/module and registry/matrix references.
The previous execution-audit result consumer remains usable.
