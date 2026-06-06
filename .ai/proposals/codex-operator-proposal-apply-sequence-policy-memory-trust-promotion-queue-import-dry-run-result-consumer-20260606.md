# Proposal: sequence-policy queue import dry-run result consumer

## Short thesis

Add a read-only consumer that verifies generic dry-run results still match
sequence-policy queue-import provenance before an execution-gate request is
exposed.

## What I understood

The sequence-policy chain now reaches a verified request for
`dry_run_operator_proposal_apply_sequence`. Once that dry-run returns, the
system needs a provenance-preserving verifier before
`gate_operator_proposal_apply_sequence_execution` is used.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py`.
It consumes the sequence-plan result consumer plus a generic dry-run payload.
It emits an execution-gate request template only when dry-run results are
`would_apply`, ordered, idempotency-safe, fader/gain bounded, and carry no-write
safety flags.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`

## Alternatives considered

Pass generic dry-run results directly to the execution gate. Rejected because
that loses sequence-policy provenance checks.

Automatically invoke the execution gate. Rejected because operator
acknowledgement must stay explicit.

## Risks

The execution-gate request template could be mistaken for executor permission.
Mitigation: the template keeps `operator_ack=false` and requires the operator to
set acknowledgement explicitly; gate invocation, executor execution, idempotency
reservation, live writes, and full autonomous readiness remain false.

## Test plan

- Focused builder tests.
- Product Layer chain test from queue import through execution-gate request.
- Registry/capability matrix tests.

## Where the other agent may disagree

Kimi may argue that generic execution gate already validates dry-run rows. The
counterargument is that generic validation does not prove the dry-run belongs to
the trusted sequence-policy import chain.
