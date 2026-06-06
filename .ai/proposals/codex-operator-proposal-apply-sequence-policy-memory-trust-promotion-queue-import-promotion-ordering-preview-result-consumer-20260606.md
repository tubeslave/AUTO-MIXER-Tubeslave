# Codex Proposal: Queue Import Promotion Ordering-Preview Result Consumer

## Short Thesis

Add a read-only verifier between promotion ordering preview and queue-ordering
planning, preserving provenance without widening queue or live authority.

## What I Understood

The project goal pushes toward autonomous sequencing and queue import, but the
current safe architecture requires each step to be proven as evidence before any
mutation. The policy-consumer result consumer already exposes an ordering-preview
request; the next missing proof is verifying that preview output before a
queue-ordering-plan request is made available.

## Proposed Solution

- Add
  `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer.py`.
- Verify the prior policy-consumer result consumer envelope, summary, and
  request template.
- Verify ordering-preview envelope, summary counts, ranked rows, unranked rows,
  row order, duplicate IDs, and safety flags.
- Emit only
  `plan_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering`
  request metadata when evidence is valid.
- Add Product Layer verifier and direct queue-ordering-plan preview handler.
- Register the method as `shadow_only` decision evidence and add matrix support.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-ordering-preview-result-consumer-20260606.md`

## Alternatives Considered

- Call the queue-ordering-plan builder directly from the policy-consumer result
  consumer. Rejected because it skips an explicit verification boundary.
- Let Product Layer accept raw ordering previews with no provenance consumer.
  Rejected because forged preview rows could imply queue authority.

## Risks

- The validator can become too strict for valid unranked proposals.
- Future schema changes in ordering preview may require updating this verifier.

## Test Plan

- Not-ready and blocked states.
- Happy path through request template and queue-ordering-plan builder.
- Forged prior consumer and forged ordering-preview rejection.
- Registry, capability matrix, and Product Layer chain tests.

## Where The Other Agent May Disagree

Kimi may argue for a broader queue import integration now. The safer step is to
keep this as a provenance consumer first, then review the acknowledged queue
import boundary separately.
