# Proposal: sequence-policy queue import execution-gate result consumer

## Short thesis

Add a read-only consumer that verifies generic execution-gate results still
match sequence-policy queue-import provenance before a sequence-executor request
is exposed.

## What I understood

The sequence-policy chain now reaches a verified request for
`gate_operator_proposal_apply_sequence_execution`. Once that gate returns, the
system needs a provenance-preserving verifier before
`execute_operator_proposal_apply_sequence` is used.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py`.
It consumes the dry-run result consumer plus a generic execution-gate payload.
It emits a sequence-executor request template only when gate rows are ready for
executor, ordered, idempotency-safe, fader/gain bounded, current-queue checked,
and carry no-write safety flags.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`

## Alternatives considered

Pass generic execution-gate rows directly to the sequence executor. Rejected
because that loses sequence-policy provenance checks.

Automatically invoke the sequence executor. Rejected because execute,
operator acknowledgement, and approval must stay explicit at the live-write
boundary.

## Risks

The executor request template could be mistaken for permission to write. The
mitigation is that the template defaults `execute=false`, `operator_ack=false`,
and `approved=false`; executor invocation, apply attempts, idempotency
reservation, queue mutation, live writes, and full autonomous readiness remain
false.

## Test plan

- Focused builder tests.
- Product Layer chain test from queue import through executor request template.
- Registry/capability matrix tests.

## Where the other agent may disagree

Kimi may argue that generic sequence executor already validates gate rows. The
counterargument is that generic validation does not prove the gate belongs to
the trusted sequence-policy import chain.
