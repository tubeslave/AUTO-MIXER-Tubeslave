# Codex Proposal: Sequence Policy Memory Trust Promotion Queue Ordering Plan

## Short thesis

Add a read-only queue-ordering plan after the sequence-policy ordering preview
so trusted sequence-memory ordering evidence can be reviewed before any queue or
execution consumer mutates state.

## What I understood

The project is building toward autonomous apply-sequence behavior. Ordering
preview records are useful evidence, but a separate queue-ordering plan is the
right next boundary before an acknowledged queue import or sequence executor.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan.py`.
It validates ordering-preview envelope, summary counters, row identity,
recommended-index sequence, duplicates, and safety flags. It emits
`queue_ordering_plan_records` as read-only candidates with operator review and
future acknowledgement requirements.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-ordering-plan-20260606.md`

## Alternatives considered

- Let ordering preview feed queue import directly. Rejected because queue import
  needs separate operator acknowledgement and target queue validation.
- Mutate the Product Layer queue order now. Rejected because current evidence is
  not authority.

## Risks

- Queue-ordering plan rows could be mistaken for import authority. Mitigation:
  keep `queue_import_allowed_by_plan=false`, `queue_reorder_allowed_by_plan=false`,
  `can_import_to_queue=false`, and `can_apply_live=false`.
- Forged preview rows could claim live or sequence authority. Mitigation: reject
  forged envelope, summary, and row-level flags before emitting plan records.

## Test plan

- Unit tests for not-ready, valid queue-ordering records, forged authority,
  duplicate rows, registry, and capability matrix.
- Extend Product Layer preview test for payload, adapter, and summary exposure.
- Run focused pytest and memory lint.

## Where the other agent may disagree

Kimi may prefer a queue importer immediately. I would keep this as an evidence
plan first because importing or reordering the queue changes runtime state and
needs a separate reviewed contract.
