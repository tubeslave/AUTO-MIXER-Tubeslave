# Codex Proposal: Promotion Apply-Audit Result Consumer

## Short Thesis

Consume promotion apply-audit output as provenance evidence and expose only a
read-only rollback-verification request after audit records match the applied
executor-result chain.

## What I Understood

The project is moving toward autonomous learning, but apply-audit records are
not rollback-verified policy evidence. Rollback verification requires a
separate operator-reviewed evidence step and must not be implied by audit
availability.

## Proposed Solution

- Add
  `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer`.
- Verify prior executor-result consumer envelope, summary, and request.
- Verify apply-audit envelope, summary, audit records, and false authority
  flags.
- Match audit records back to executor apply results by apply id, policy key,
  target path, rollback target state, and idempotency key.
- Emit a read-only
  `verify_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback`
  request that still requires operator verification records.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- focused tests and Product Layer flow tests
- `.ai/briefs`, `.ai/proposals`, `.ai/memory/project.md`, `Docs/adr`

## Alternatives Considered

- Let the executor-result consumer emit rollback-verification requests
  directly. Rejected because apply-audit output needs its own validation.
- Let this consumer call rollback verification directly. Rejected because
  verification records are operator-reviewed evidence and must remain a
  separate step.

## Risks

- Audit records could be mistaken for rollback-verified policy evidence. The
  consumer emits only a template and keeps rollback verification false.
- Forged audit records could claim authority. The verifier checks record-level
  identity, rollback flags, counts, target paths, and false live/queue/model
  flags.

## Test Plan

Run focused pytest for the new consumer, Product Layer path, registry/matrix
tests, memory lint, diff whitespace checks, promotion-chain pytest, and backend
suite when practical.

## Where The Other Agent May Disagree

Kimi may argue that rollback verification should stay entirely manual. This
consumer preserves that boundary: it only prepares a request template and
requires separate operator-supplied verification records.
