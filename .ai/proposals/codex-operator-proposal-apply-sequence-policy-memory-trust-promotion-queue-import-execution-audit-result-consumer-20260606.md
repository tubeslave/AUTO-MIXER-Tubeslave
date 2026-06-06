# Proposal: sequence-policy queue import execution-audit result consumer

## Short thesis

Add a read-only consumer that verifies generic execution-audit results still
match sequence-policy queue-import provenance before an audit-policy-gate
request is exposed.

## What I understood

The sequence-policy chain now reaches a verified request for
`audit_operator_proposal_apply_sequence_execution`. Once that audit returns, the
system needs a provenance-preserving verifier before
`gate_operator_proposal_apply_sequence_audit_policy` is used.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py`.
It consumes the execution-result consumer plus a generic execution-audit
payload. It emits an audit-policy-gate request template only when audit rows are
passed, ordered, idempotency-safe, fader/gain bounded, readback/proof verified,
rollback-capable, and no-write safety flags are intact.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`

## Alternatives considered

Pass generic execution-audit results directly to the audit-policy gate. Rejected
because that loses sequence-policy provenance checks.

Automatically invoke the audit-policy gate. Rejected because this consumer is an
evidence bridge only and policy-gate invocation must remain a separate handler.

## Risks

The audit-policy-gate request template could be mistaken for policy mutation
authority. Mitigation: the consumer keeps policy gate invocation, policy
mutation, runtime-memory mutation, model mutation, queue mutation, live writes,
live authority, and full autonomous readiness false.

## Test plan

- Focused builder tests.
- Product Layer chain test from queue import through audit-policy-gate request.
- Registry/capability matrix tests.

## Where the other agent may disagree

Kimi may argue that generic audit-policy gate already validates audit rows. The
counterargument is that generic validation does not prove the audit belongs to
the trusted sequence-policy import chain.
