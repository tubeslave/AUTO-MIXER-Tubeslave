# Codex Proposal: Promotion Audit Policy Gate Result Consumer

## Short Thesis

Add a thin, read-only promotion wrapper around the generic audit-policy-gate result consumer so verified promotion policy evidence can advance to a memory-draft request template without invoking memory draft or live-write behavior.

## What I Understood

The promotion chain needs the same staged evidence barriers as the generic sequence-policy queue-import path. The previous consumer stops at an audit-policy-gate request; this step should consume the returned gate result and expose the next request only after provenance, ids, idempotency keys, fader/gain bounds, readback/proof, and rollback evidence still match.

## Proposed Solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer`.
- Validate the promotion execution-audit consumer envelope, status, summary, request template, UTC timestamp, and false safety flags.
- Delegate generic policy evidence validation to `build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer`.
- Emit `promotion_memory_draft_request` only when the generic consumer returns `sequence_memory_trust_promotion_queue_import_memory_draft_request_available`.
- Keep memory-draft invocation, memory creation, memory writes, policy/runtime/model/queue mutation, idempotency reservation, direct sequence execution, live authority, and full autonomous readiness false.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-audit-policy-gate-result-consumer-20260606.md`

## Alternatives Considered

- Call the memory-draft handler directly from the promotion consumer. Rejected because it would merge verification and action boundaries.
- Duplicate generic policy-evidence validation. Rejected because the generic consumer is already the canonical validator for that layer.

## Risks

- Forged policy evidence could bypass promotion provenance if the wrapper validates only generic output.
- The consumer could be misread as memory-write permission unless request and safety flags stay explicit.

## Test Plan

- Focused pytest for not-ready, ready, forged, blocked, registry, and matrix behavior.
- Product State dispatch coverage for promotion execution-audit result to promotion memory-draft request.
- `py_compile`, memory lint, `git diff --check`, and focused pytest.

## Where The Other Agent May Disagree

Kimi may argue this chain is too granular. The counterpoint is that live-sound memory promotion benefits from explicit, inspectable gates with independent rollback and safety evidence.
