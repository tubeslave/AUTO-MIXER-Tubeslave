# Codex Proposal: Promotion Sequence-Plan Result Consumer

## 1. Short thesis

Add a read-only verifier that accepts a generic sequence-plan result only when
it matches promotion batch-preflight provenance, then emits a dry-run request
template without executing dry-run or widening live authority.

## 2. What I understood

Automixer is building a long supervised-promotion evidence chain. Each step must
prove the previous step and only expose the next request template. The current
open gap is between a verified promotion sequence-plan request and any later
dry-run evidence.

## 3. Proposed solution

Create `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer`.
It delegates generic sequence-plan validation to the existing generic
sequence-plan result consumer, but also validates the promotion batch-preflight
consumer envelope, summary, and request template.

## 4. Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-sequence-plan-result-consumer-20260606.md`

## 5. Alternatives considered

The shortest alternative was to reuse the generic sequence-plan result consumer
directly. That loses promotion-chain provenance, so future code could confuse a
generic sequence plan with a promotion-verified sequence plan.

## 6. Risks

The method chain is verbose. The mitigation is to keep the consumer small,
transport-free, and covered by tests that assert every mutation/live-authority
flag remains false.

## 7. Test plan

- Compile the new consumer and touched integration files.
- Run focused consumer tests.
- Run the product-state handler chain test.
- Run registry and project-goal matrix tests.
- Run full pytest and memory lint.

## 8. Where the other agent may disagree

Kimi might prefer consolidating the promotion result consumers into one generic
adapter. For this repo state, a separate method keeps the proof boundary
auditable and avoids a broader refactor.
