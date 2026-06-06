# Codex Proposal: Promotion Execution-Plan Result Consumer

## Short Thesis

Add the narrow consumer as a read-only provenance verifier that exposes an
executor dry-run request template, not as a dry-run runner or policy executor.

## What I Understood

The system is moving toward autonomous sequence-policy memory promotion, but
each boundary must remain explicit. Accepted promotion records may become
execution-plan candidates, and those candidates may be dry-run, but no consumer
in this slice may apply policy or touch live console state.

## Proposed Solution

- Register `operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer`.
- Verify a prior promotion-acceptance-ledger result consumer and a promotion
  execution-plan result.
- Emit `promotion_executor_dry_run_request` with
  `dry_run_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor`
  as the handler and `apply=False`.
- Add Product Layer handlers for verifying the plan result and invoking the
  existing executor dry-run builder.
- Track the method in the registry, project-goal matrix, and durable memory.

## Likely Files To Touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_product_state.py`
- focused tests for the new consumer
- `.ai/briefs`, `.ai/proposals`, `.ai/memory/project.md`, `Docs/adr`

## Alternatives Considered

- Invoke the executor dry-run directly inside the consumer. Rejected because it
  collapses two review boundaries.
- Treat the execution plan as policy promotion authority. Rejected because this
  would widen memory/policy influence without an acknowledged executor contract.

## Risks

- Handler naming could imply real execution. The selected dry-run handler name
  includes `dry_run` and keeps `apply` false by default.
- Forged plans could claim authority. Tests mutate safety flags, summary counts,
  and candidate identity to ensure replacement is required.

## Test Plan

Run focused pytest for the new consumer, Product Layer path, and capability
matrix, plus py_compile, memory lint, and diff whitespace checks.

## Where The Other Agent May Disagree

Kimi may argue for jumping directly to an executor contract. That is premature
until the queue-import execution-plan provenance boundary is separately tested.
