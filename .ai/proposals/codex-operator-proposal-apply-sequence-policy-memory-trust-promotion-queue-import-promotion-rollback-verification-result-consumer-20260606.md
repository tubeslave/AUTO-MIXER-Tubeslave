# Codex Proposal: Promotion Rollback-Verification Result Consumer

## Short Thesis

Consume rollback-verification output as provenance evidence and expose only a
read-only sequence-policy gate request after verified records match the
apply-audit chain.

## What I Understood

The project is moving toward autonomous sequence-policy learning, but verified
rollback evidence is not itself policy-gate output. The sequence-policy gate is
a separate read-only step and still does not mutate policy stores, queues, or
live mixer state.

## Proposed Solution

- Add
  `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer`.
- Verify prior apply-audit result consumer envelope, summary, and request.
- Verify rollback-verification envelope, summary, verified records, and false
  authority flags.
- Match verified rollback records back to apply-audit records by audit id,
  proposal, policy key, target path, rollback state, and evidence status.
- Emit a read-only
  `gate_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy`
  request.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- focused tests and Product Layer flow tests
- `.ai/briefs`, `.ai/proposals`, `.ai/memory/project.md`, `Docs/adr`

## Alternatives Considered

- Let the apply-audit result consumer emit sequence-policy gate requests
  directly. Rejected because rollback verification needs its own validation.
- Let this consumer call the sequence-policy gate directly. Rejected because
  result consumption and gate construction should remain separately reviewable.

## Risks

- Verified rollback records could be mistaken for policy mutation authority.
  The consumer emits only a request template and keeps policy mutation false.
- Forged verification records could claim live or queue authority. The verifier
  checks identity, target paths, evidence status, counts, and false live/queue
  flags.

## Test Plan

Run focused pytest for the new consumer, Product Layer path, registry/matrix
tests, memory lint, diff whitespace checks, promotion-chain pytest, and backend
suite when practical.

## Where The Other Agent May Disagree

Kimi may argue that sequence-policy gate requests should be fully manual. This
consumer preserves that boundary: it only prepares a request template after
verified rollback records match provenance and does not create policy records.
