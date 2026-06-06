# Proposal: sequence-policy queue import preflight result consumer

## Short thesis

Add a read-only consumer that verifies approved batch-preflight results against
the sequence-policy queue-import preflight plan before the generic sequence-plan
handler is used.

## What I understood

The existing bridge can select pending sequence-policy imported proposals and
emit a batch-preflight request template. The next autonomy step needs stronger
evidence that the batch result was not reordered, forged, widened, or detached
from the sequence-policy import before sequence planning.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py`.
It consumes a preflight plan plus a batch preflight payload and returns a
sequence-plan request template only when every planned preflight candidate has a
matching ready result in the same order, with expected idempotency keys and
fail-closed safety flags.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`

## Alternatives considered

Automatically call `plan_operator_proposal_apply_sequence` from the preflight
plan handler. Rejected because that would blur operator action boundaries and
make sequence planning implicit.

Reuse the generic sequence-plan handler directly. Rejected as incomplete because
it does not know whether the batch preflight result came from the
sequence-policy queue-import template.

## Risks

The verifier could be mistaken for execution authority. Mitigation: it returns
only a request template and keeps `sequence_plan_created=false`,
`sequence_execution_allowed_by_consumer=false`, `idempotency_reserved=false`,
`live_write=false`, and `full_autonomous_ready=false`.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py -q`
- Product Layer chain test from queue import through generic sequence plan.
- Registry and capability-matrix tests.

## Where the other agent may disagree

Kimi may argue that this is too many small gates. The counterargument is that
this gate closes a real provenance gap between trusted sequence-policy ordering
and generic apply-sequence planning without expanding live authority.
