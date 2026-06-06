# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-audit-policy-gate-result-consumer-20260606

## Short thesis

Add a read-only result consumer between generic audit-policy gate output and
generic policy-memory draft input, preserving sequence-policy queue-import
provenance without granting memory-write or live authority.

## What I understood

The queue-import path is meant to reuse generic apply-sequence handlers while
keeping every boundary explicit. Execution audit can now feed a generic
audit-policy gate. The policy evidence from that gate still needs a
sequence-policy-specific verification layer before any memory draft is exposed.

## Proposed solution

- Build
  `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer`.
- Require the upstream execution-audit result consumer to be available and
  read-only.
- Require a generic `operator_proposal_apply_sequence_audit_policy_gate` with
  ready `policy_evidence_ready` rows, verified readback/proof, rollback,
  fader/gain bounds, matching idempotency keys, and no live or mutation flags.
- Emit only a `draft_operator_proposal_apply_sequence_policy_memory` request
  template.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-audit-policy-gate-result-consumer-20260606.md`

## Alternatives considered

Directly passing the audit-policy gate result to the generic memory draft
handler is simpler, but loses the queue-import provenance check. Calling the
memory draft handler automatically is also simpler for the UI flow, but widens
authority across a boundary that should remain explicit.

## Risks

The main risk is that a memory-draft request is mistaken for memory mutation or
autonomous readiness. The payload must keep draft invocation, memory mutation,
policy mutation, runtime-memory mutation, model mutation, queue mutation, live
authority, and full autonomous readiness false.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py backend/handlers/product_state_handlers.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py tests/test_operator_product_state.py::test_sequence_policy_queue_import_execution_result_consumer_feeds_audit_without_extra_write tests/test_project_goal_capability_matrix.py -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Kimi may argue that memory draft should be automatically created after the
policy gate. The safer boundary is to return a verified request template only;
the generic memory-draft handler remains a separate explicit step.
