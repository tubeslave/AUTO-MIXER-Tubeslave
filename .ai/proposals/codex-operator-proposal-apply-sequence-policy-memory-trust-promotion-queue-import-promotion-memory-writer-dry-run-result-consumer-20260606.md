# Codex Proposal: Promotion Memory Writer Dry Run Result Consumer

## Short Thesis

Add a thin, read-only promotion wrapper around the generic memory-writer dry-run result consumer so verified promotion memory-writer dry-run evidence can advance to an explicit writer apply request template without writing memory.

## What I Understood

The promotion chain now reaches a writer request template. The generic writer dry-run intentionally blocks persistence until explicit operator acknowledgement and apply are supplied. The promotion wrapper should preserve that dry-run blocker and expose only a future apply template.

## Proposed Solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer`.
- Validate the promotion memory-draft result consumer envelope, status, summary, request template, UTC timestamp, and false safety flags.
- Delegate generic dry-run validation to `build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer`.
- Emit `promotion_memory_writer_apply_request` only when the generic consumer returns `sequence_memory_trust_promotion_queue_import_memory_writer_apply_request_available`.
- Keep writer invocation, memory file writes, policy/runtime/model/queue mutation, idempotency reservation, direct sequence execution, live authority, and full autonomous readiness false.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-memory-writer-dry-run-result-consumer-20260606.md`

## Alternatives Considered

- Call the memory writer apply path directly from the promotion consumer. Rejected because it would merge dry-run verification with persistence.
- Duplicate generic dry-run validation. Rejected because the generic consumer is already the canonical validator for dry-run writer evidence.

## Risks

- A forged dry-run could be treated as write readiness if the expected dry-run blocker is not preserved.
- Operators could mistake the apply template for permission unless `apply=false`, `operator_ack=false`, and false mutation flags remain explicit.

## Test Plan

- Focused pytest for not-ready, ready, forged, applied-result rejection, registry, and matrix behavior.
- Product State dispatch coverage for promotion memory-writer dry-run result to promotion apply request.
- `py_compile`, memory lint, `git diff --check`, and focused/full pytest.

## Where The Other Agent May Disagree

Kimi may argue this is excessive layering. The counterpoint is that memory persistence affects future policy decisions, so dry-run evidence and explicit write authorization should remain separate.
