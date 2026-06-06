# Codex Proposal: Promotion Executor Result Consumer

## Short Thesis

Consume promotion executor output as provenance evidence and expose only a
read-only apply-audit request after the executor has applied to an explicit
target policy store.

## What I Understood

The project is moving toward autonomous learning, but executor preview output is
not audit-ready. Audit should only be available after the separate executor call
has explicit `apply=True`, `operator_ack=True`, and a target policy store.

## Proposed Solution

- Add
  `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer`.
- Verify prior executor dry-run result consumer envelope, summary, and request.
- Verify executor output, blocking preview-only results and accepting only
  `sequence_memory_trust_promotion_executor_applied`.
- Match apply results back to dry-run specs by dry-run id, policy key, target
  path, idempotency key, and false live/queue/model flags.
- Emit a read-only
  `audit_operator_proposal_apply_sequence_policy_memory_trust_promotion` request.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- focused tests and Product Layer flow tests
- `.ai/briefs`, `.ai/proposals`, `.ai/memory/project.md`, `Docs/adr`

## Alternatives Considered

- Let the dry-run result consumer emit audit requests directly. Rejected because
  preview and applied executor results need different treatment.
- Let this consumer call apply audit directly. Rejected because result
  consumption and audit construction should remain separate reviewable steps.

## Risks

- A preview executor result could be mistaken for applied policy evidence. The
  consumer blocks preview-only `sequence_memory_trust_promotion_executor_ready`.
- Forged apply results could claim policy-store mutation. The verifier checks
  dry-run ids, policy keys, target paths, idempotency keys, and false authority
  flags.

## Test Plan

Run focused pytest for the new consumer, Product Layer path, registry/matrix
tests, memory lint, diff whitespace checks, promotion-chain pytest, and backend
suite.

## Where The Other Agent May Disagree

Kimi may argue that applied policy-store mutation should wait. This consumer
does not apply; it only verifies a separate explicit executor result and keeps
audit as a separate read-only request.
