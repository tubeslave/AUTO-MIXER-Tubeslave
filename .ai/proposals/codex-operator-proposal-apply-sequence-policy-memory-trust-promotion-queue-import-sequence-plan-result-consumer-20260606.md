# Proposal: sequence-policy queue import sequence-plan result consumer

## Short thesis

Add a read-only consumer that verifies generic sequence-plan results still match
sequence-policy queue-import provenance before a dry-run request is exposed.

## What I understood

The sequence-policy path now reaches a verified request for
`plan_operator_proposal_apply_sequence`. Once that generic handler returns a
sequence plan, the system needs a provenance-preserving verifier before the
plan is handed to `dry_run_operator_proposal_apply_sequence`.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py`.
It consumes the preflight-result consumer payload plus the generic sequence-plan
payload. It emits a dry-run request template only when planned proposal IDs,
idempotency keys, step safety flags, fader/gain constraints, rollback fields,
and no-write summary flags match the trusted sequence-policy chain.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`

## Alternatives considered

Pass generic sequence plans directly to dry-run. Rejected because that loses the
sequence-policy import provenance check.

Automatically invoke dry-run from the verifier. Rejected because dry-run should
remain an explicit operator/review step.

## Risks

The dry-run request template could be misunderstood as execution permission.
Mitigation: keep `dry_run_handler_invoked=false`, `dry_run_created=false`,
`execution_allowed_by_consumer=false`, `idempotency_reserved=false`,
`live_write=false`, and `full_autonomous_ready=false`.

## Test plan

- Focused builder tests.
- Product Layer chain test from queue import through dry-run request.
- Registry/capability matrix tests.

## Where the other agent may disagree

Kimi may argue the generic dry-run handler already validates sequence plans.
The counterargument is that generic validation does not prove the plan still
belongs to the trusted sequence-policy import chain.
