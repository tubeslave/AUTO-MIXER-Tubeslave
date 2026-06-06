# Proposal: sequence-policy queue import execution-result consumer

## Short thesis

Add a read-only consumer that verifies generic executor results still match
sequence-policy queue-import provenance before an execution-audit request is
exposed.

## What I understood

The sequence-policy chain now reaches a verified request for
`execute_operator_proposal_apply_sequence`. Once that executor returns, the
system needs a provenance-preserving verifier before
`audit_operator_proposal_apply_sequence_execution` is used.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py`.
It consumes the execution-gate result consumer plus a generic executor result.
It emits an execution-audit request template only when executor rows are
applied, ordered, idempotency-safe, delegated through `apply_operator_proposal`,
verified by readback/proof evidence, rollback-capable, and fader/gain bounded.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`

## Alternatives considered

Pass generic executor results directly to the audit handler. Rejected because
that loses sequence-policy provenance checks.

Automatically invoke the audit handler. Rejected because the consumer should be
an evidence bridge only; audit remains a separate explicit handler.

## Risks

The consumer observes executor evidence that may include real applied writes.
Mitigation: consumer-level safety flags keep audit invocation, apply attempts,
idempotency reservation, queue mutation, live writes, live authority, and full
autonomous readiness false.

## Test plan

- Focused builder tests.
- Product Layer chain test from queue import through audit request template.
- Registry/capability matrix tests.

## Where the other agent may disagree

Kimi may argue that generic execution audit already validates executor rows.
The counterargument is that generic audit validation does not prove the result
belongs to the trusted sequence-policy import chain.
