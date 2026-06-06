# Proposal: execution-result consumer for promotion queue import

## Short thesis

Add a read-only result consumer between the promotion sequence-executor request and the generic execution-audit handler. It should verify promotion provenance and delegated apply evidence, then emit only an audit request template.

## What I understood

The promotion chain can now expose a generic executor request template. Once an operator runs that executor with explicit execute, acknowledgement, and approval, the result still needs promotion provenance verification before audit evidence can be requested.

## Proposed solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer`.
- Verify the prior promotion execution-gate result consumer envelope, safety flags, ready proposal ids, idempotency prefix, and emitted executor request.
- Reuse the existing generic queue-import execution-result consumer to validate executor output.
- Reject partial/blocked rows, forged direct live-write authority, direct manual console-write claims, idempotency mismatch, unsafe fader values, and mismatched provenance.
- Emit only an `audit_operator_proposal_apply_sequence_execution` request template.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Feed executor results directly into the generic audit handler. Rejected because it bypasses promotion provenance checks.
- Duplicate generic execution-result validation. Rejected because the existing queue-import execution-result consumer owns the delegated-apply contract.
- Invoke audit inside this consumer. Rejected because result consumers should expose request templates, not execute the next Product Layer step.

## Risks

- The wrapper depends on the generic execution-result schema. Mitigation: invalid generic evidence is surfaced under `generic_execution_result_consumer:*` invalid ids and focused tests cover forged paths.
- The source execution result reflects a live delegated apply. Mitigation: this consumer itself remains read-only, only accepting evidence already validated by the generic consumer.

## Test plan

- Focused pytest for the new consumer.
- Product Layer dispatch pytest through audit request exposure.
- Registry and capability matrix checks.
- `py_compile`, `memory_lint`, and `git diff --check`.

## Where the other agent may disagree

Kimi might prefer this wrapper to reject all execution results with `live_write=true`. I would allow only generic-consumer-validated delegated `apply_operator_proposal` evidence, because the next audit stage must inspect the actual supervised apply result.
