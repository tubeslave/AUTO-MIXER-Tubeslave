# Codex Proposal: Queue Import Promotion Queue-Ordering-Plan Result Consumer

## Short Thesis

Add a read-only verifier between queue-ordering planning and queue importer
dry-run, preserving provenance without mutating the operator queue.

## What I Understood

The project needs autonomous queue import eventually, but queue mutation is a
high-trust boundary. The ordering-preview result consumer can expose a
queue-ordering-plan request. The next missing proof is verifying that plan
before a queue-importer request is made available.

## Proposed Solution

- Add
  `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer.py`.
- Verify the prior ordering-preview result consumer envelope, summary, and
  request template.
- Verify queue-ordering-plan envelope, summary counts, record order, duplicate
  IDs, record provenance, and safety flags.
- Emit only
  `import_operator_proposal_apply_sequence_policy_queue_ordering_plan` request
  metadata with `apply=false`, `operator_ack=false`, and
  `proposal_payloads_required_from_operator=true`.
- Add Product Layer verifier wiring.
- Register the method as `shadow_only` decision evidence and add matrix support.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-promotion-queue-ordering-plan-result-consumer-20260606.md`

## Alternatives Considered

- Invoke the queue importer directly from the queue-ordering-plan handler.
  Rejected because queue import requires a distinct operator-gated boundary.
- Let Product Layer accept raw queue-ordering plans for importer requests.
  Rejected because forged plan records could imply queue authority.

## Risks

- The validator can become too strict if queue-ordering-plan record schema
  evolves.
- The emitted request still needs proposal payloads from the operator context,
  so downstream UI must supply them deliberately.

## Test Plan

- Not-ready and blocked states.
- Happy path through request template and queue-importer dry-run.
- Forged prior consumer and forged queue-ordering-plan rejection.
- Registry, capability matrix, and Product Layer chain tests.

## Where The Other Agent May Disagree

Kimi may prefer moving directly to acknowledged queue import. The safer step is
to prove queue-ordering-plan provenance first and keep importer invocation as a
separate dry-run/operator-ack boundary.
