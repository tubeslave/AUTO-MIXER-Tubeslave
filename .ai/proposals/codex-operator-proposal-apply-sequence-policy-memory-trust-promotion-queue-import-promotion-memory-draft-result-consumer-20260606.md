# Codex Proposal: Promotion Memory Draft Result Consumer

## Short Thesis

Add a thin, read-only promotion wrapper around the generic memory-draft result consumer so verified promotion memory drafts can advance to a memory-writer request template without writing memory or widening live authority.

## What I Understood

The promotion chain now reaches a memory-draft request template. The next step should consume the returned draft result, verify draft records and evidence against promotion provenance, and emit the writer request only as a template requiring separate operator acknowledgement and apply.

## Proposed Solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer`.
- Validate the promotion audit-policy-gate consumer envelope, status, summary, request template, UTC timestamp, and false safety flags.
- Delegate generic memory-draft validation to `build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer`.
- Emit `promotion_memory_writer_request` only when the generic consumer returns `sequence_memory_trust_promotion_queue_import_memory_writer_request_available`.
- Keep writer invocation, memory file writes, policy/runtime/model/queue mutation, idempotency reservation, direct sequence execution, live authority, and full autonomous readiness false.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-memory-draft-result-consumer-20260606.md`

## Alternatives Considered

- Call the memory writer directly from the promotion consumer. Rejected because it would merge verification with persistence.
- Duplicate generic draft-record validation. Rejected because the generic consumer is already the canonical validator for this layer.

## Risks

- A forged memory draft could be treated as writer input if promotion provenance is not checked.
- Operators could mistake a writer request template for write permission unless `apply=false`, `operator_ack=false`, and false mutation flags remain explicit.

## Test Plan

- Focused pytest for not-ready, ready, forged, blocked, registry, and matrix behavior.
- Product State dispatch coverage for promotion memory-draft result to promotion memory-writer request.
- `py_compile`, memory lint, `git diff --check`, and focused/full pytest.

## Where The Other Agent May Disagree

Kimi may prefer a shorter chain. The counterpoint is that memory persistence affects future decisions, so draft verification and writer execution should remain separately reviewable.
