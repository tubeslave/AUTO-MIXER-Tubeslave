# Proposal: queue-importer result consumer for promotion queue import

## Short thesis

Add a read-only result consumer between the promotion queue importer and the existing queue-import preflight planner. It must require applied queue-import evidence and keep dry-run importer output blocked.

## What I understood

The current chain can verify queue-ordering-plan rows and expose a queue-importer request. The importer itself is operator-gated: dry-run is non-mutating, while `apply=true` plus `operator_ack=true` mutates only an explicit operator proposal queue. The next safe step is to verify that importer result before preflight planning.

## Proposed solution

- Add `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer`.
- Verify the prior queue-ordering-plan result consumer envelope and request template.
- Verify queue-importer envelope, specs, applied result rows, and imported proposal metadata.
- Block dry-run importer results before preflight planning.
- Emit only a `plan_operator_proposal_apply_sequence_policy_queue_import_preflight` request template after an applied importer result.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`

## Alternatives considered

- Let the queue-ordering-plan consumer emit a preflight request directly. Rejected because dry-run importer output is not queue state and preflight planning needs actual pending queue actions.
- Let this consumer call the preflight handler. Rejected because that would combine verification with execution of the next Product Layer handler.

## Risks

- The verifier may be too strict if importer result rows evolve. Mitigation: checks are scoped to current importer contract and focused tests cover the intended path.
- Applied queue import mutates the operator proposal queue. Mitigation: this consumer does not apply import; it only validates a separately operator-acknowledged result.

## Test plan

- Focused pytest for the new consumer.
- Product Layer dispatch pytest for applied import -> verifier -> preflight plan.
- Registry and capability matrix checks.
- `py_compile`, `memory_lint`, `git diff --check`.

## Where the other agent may disagree

Kimi might prefer to require this consumer to inspect live queue contents directly. I would keep that out of the verifier: the existing preflight-plan handler already reads the authoritative operator queue, and this layer should remain provenance validation plus request templating only.
