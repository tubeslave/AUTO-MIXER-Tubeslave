# Codex Proposal: Sequence Policy Queue Import Preflight Plan

## Short thesis

Add a read-only preflight planning bridge from sequence-policy imported queue
proposals to the existing batch preflight handler.

## What I understood

Queue import moves trusted sequence-memory proposals into the operator queue,
but execution still requires the existing preflight, sequence plan, dry-run,
execution gate, and executor path. The missing link is a deterministic way to
select the imported proposals in their trusted order for batch preflight.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan.py`.
It reads pending actions, filters sequence-policy imported proposals, validates
their import metadata, sorts by planned queue index, and emits a
`preflight_operator_proposal_apply_batch` request template with proposal IDs and
idempotency prefix. It does not invoke the handler itself.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan.py`
- `backend/operator_product_state.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-preflight-plan-20260606.md`

## Alternatives considered

- Automatically run batch preflight. Rejected because the bridge should remain
  a request template and not perform operator approval.
- Jump directly to sequence planning. Rejected because current sequence plans
  require fresh preflight evidence.

## Risks

- The request template could be mistaken for approval. Mitigation: set
  `approved=false`, require the operator to set approval explicitly, and keep
  idempotency reservation false.
- Forged import metadata could remove the separate sequence-consumer boundary.
  Mitigation: reject metadata where sequence consumer and live safety flags do
  not remain fail-closed.

## Test plan

- Unit-test not-ready, valid ordering, analysis-only skip, forged metadata,
  registry, and capability matrix.
- Extend Product State tests for decision-queue visibility and handler-to-batch
  preflight compatibility.
- Run focused pytest, memory lint, and diff checks.

## Where the other agent may disagree

Kimi may prefer running preflight automatically from this handler. I would keep
the module as a plan producer because preflight readiness already encodes
operator approval, mixer connectivity, and idempotency evidence.
