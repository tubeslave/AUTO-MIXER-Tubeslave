# Proposal: execution-gate result consumer for promotion queue import

## Short thesis

Add a read-only result consumer between the promotion execution-gate request and the generic sequence executor. It should verify promotion provenance and gate-ready rows, then emit only an executor request template.

## What I understood

The promotion chain can now request and run a generic execution gate. That gate is readiness evidence, not executor authority. A separate promotion verifier should ensure the gate result matches prior promotion dry-run evidence before any executor request is exposed.

## Proposed solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer`.
- Verify the prior promotion dry-run result consumer envelope, safety flags, proposal ids, idempotency prefix, and emitted gate request.
- Reuse the existing generic queue-import execution-gate result consumer to validate the gate payload itself.
- Reject blocked rows, forged live-write flags, idempotency mismatch, unsafe fader values, and mismatched provenance.
- Emit only an `execute_operator_proposal_apply_sequence` request template with `execute=false`, `operator_ack=false`, and `approved=false`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Let the promotion dry-run result consumer emit an executor request directly. Rejected because execution-gate evidence must be verified first.
- Duplicate generic execution-gate validation. Rejected because the existing queue-import execution-gate result consumer owns that contract.
- Invoke the executor from inside this consumer. Rejected because result consumers should expose request templates, not execute the next Product Layer step.

## Risks

- The wrapper depends on the generic execution-gate result schema. Mitigation: invalid generic evidence is surfaced under `generic_execution_gate_result_consumer:*` invalid ids and focused tests cover forged paths.
- The executor request can look close to live execution. Mitigation: it keeps `execute=false`, `operator_ack=false`, and `approved=false`; the generic executor remains a separate supervised action.

## Test plan

- Focused pytest for the new consumer.
- Product Layer dispatch pytest through executor request exposure.
- Registry and capability matrix checks.
- `py_compile`, `memory_lint`, `git diff --check`, and full pytest.

## Where the other agent may disagree

Kimi might prefer the promotion verifier to re-parse every gate row itself. I would delegate row-level validation to the generic verifier to avoid split safety rules.
