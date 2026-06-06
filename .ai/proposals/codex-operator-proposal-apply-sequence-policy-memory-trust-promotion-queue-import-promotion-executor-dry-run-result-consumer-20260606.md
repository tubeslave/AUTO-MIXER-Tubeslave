# Codex Proposal: Promotion Executor Dry-Run Result Consumer

## Short Thesis

Consume promotion executor dry-run evidence as read-only provenance, then expose
an executor request template for a separate operator-gated step.

## What I Understood

The project is intentionally moving toward autonomous sequence-policy learning,
but dry-run evidence is not policy mutation authority. The queue-import chain
needs one more narrow verifier before the existing executor can be previewed or,
later, explicitly applied to a supplied policy store.

## Proposed Solution

- Add
  `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer`.
- Verify the prior promotion execution-plan result consumer envelope, summary,
  request, and dry-run records.
- Compare dry-run records back to execution-plan candidates by candidate id,
  proposal id, source idempotency key, target policy path, and safety flags.
- Emit `promotion_executor_request` for
  `execute_operator_proposal_apply_sequence_policy_memory_trust_promotion` with
  `apply=False` and `operator_ack=False`.
- Register the method as shadow-only decision evidence and document the boundary.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- focused tests and Product Layer flow tests
- `.ai/briefs`, `.ai/proposals`, `.ai/memory/project.md`, `Docs/adr`

## Alternatives Considered

- Invoke the promotion executor directly from the consumer. Rejected because it
  would collapse the dry-run review boundary.
- Allow the consumer to apply policy with operator acknowledgement. Rejected
  because this slice is evidence-only; apply must remain a separate explicit
  request with a supplied target policy store.

## Risks

- The executor request template could be misread as permission to mutate policy.
  The payload keeps `apply=False`, `operator_ack=False`, and explicit
  mutation flags false.
- Forged dry-run records could claim safe policy paths. The verifier checks
  target paths, idempotency keys, source candidate identity, and false authority
  flags.

## Test Plan

Run focused pytest for the new consumer, Product Layer path, registry/matrix
tests, memory lint, diff whitespace checks, and the standard backend suite.

## Where The Other Agent May Disagree

Kimi may push for applying to a real policy store now. That should wait until
this result-consumer boundary is independently proven and reviewed.
