# Proposal: execution-audit result consumer for promotion queue import

## Short thesis

Add a read-only result consumer between the promotion execution-audit request and the generic audit-policy gate. It should verify promotion provenance and passed audit rows, then emit only an audit-policy-gate request template.

## What I understood

The promotion chain can now expose a generic execution-audit request template. Once a generic audit result exists, the promotion path still needs provenance checks before exposing the policy gate that turns audit rows into policy evidence.

## Proposed solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer`.
- Verify the prior promotion execution-result consumer envelope, safety flags, proposal ids, idempotency prefix, and emitted audit request.
- Reuse the existing generic queue-import execution-audit result consumer to validate audit rows.
- Reject blocked audit rows, forged live authority, widened write kinds, idempotency mismatch, unsafe fader values, and mismatched provenance.
- Emit only a `gate_operator_proposal_apply_sequence_audit_policy` request template.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Feed execution-audit results directly into the generic policy gate. Rejected because it bypasses promotion provenance checks.
- Duplicate generic audit validation. Rejected because the existing queue-import execution-audit result consumer owns that contract.
- Invoke the policy gate inside this consumer. Rejected because result consumers should expose request templates, not execute the next Product Layer step.

## Risks

- The wrapper depends on the generic execution-audit schema. Mitigation: invalid generic evidence is surfaced under `generic_execution_audit_result_consumer:*` invalid ids and focused tests cover forged paths.
- The policy-gate request is close to policy learning. Mitigation: this consumer keeps policy mutation false; later gate, memory-draft, writer, review, and acceptance stages remain separate.

## Test plan

- Focused pytest for the new consumer.
- Product Layer dispatch pytest through audit-policy-gate request exposure.
- Registry and capability matrix checks.
- `py_compile`, `memory_lint`, and `git diff --check`.

## Where the other agent may disagree

Kimi might prefer direct audit-to-policy-gate once generic audit passes. I would keep this wrapper because the promotion chain needs explicit provenance before any policy evidence path.
