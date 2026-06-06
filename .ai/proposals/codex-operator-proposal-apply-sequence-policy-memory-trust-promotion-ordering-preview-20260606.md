# Codex Proposal: Sequence Policy Memory Trust Promotion Ordering Preview

## Short thesis

Add a read-only ordering preview after the sequence-policy consumer so trusted
sequence-memory evidence can recommend proposal order without mutating queues,
proposal payloads, policy stores, or live state.

## What I understood

The project needs increasingly autonomous apply-sequence behavior, but each
promotion boundary must stay explicit. The policy consumer can rank evidence;
ordering behavior should first appear as a separate preview before any queue or
execution consumer can use it.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview.py`.
It validates the policy-consumer envelope, summary counters, record identity,
duplicates, and safety flags. It emits `ordered_proposals` as read-only evidence,
putting sequence-policy-ranked proposals first and leaving unranked proposals in
original order.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-ordering-preview-20260606.md`

## Alternatives considered

- Feed policy-consumer records directly to queue import or sequence execution.
  Rejected because queue and execution behavior need separate reviewed
  contracts.
- Mutate the proposal order in Product Layer preview. Rejected because preview
  must remain observational and replayable.

## Risks

- Operators or future code could mistake preview rows for queue import authority.
  Mitigation: keep `proposal_order_mutation=false`, `queue_mutation=false`,
  `sequence_execution_allowed_by_ordering_preview=false`, `can_import_to_queue=false`,
  and `can_apply_live=false`.
- Forged consumer evidence could claim live or execution authority. Mitigation:
  reject forged envelope, summary, and record-level flags before producing rows.

## Test plan

- Unit tests for not-ready, recommended order, forged authority, duplicate
  records, registry, and capability matrix.
- Extend Product Layer preview test for payload, adapter, and summary exposure.
- Run focused pytest and memory lint.

## Where the other agent may disagree

Kimi may prefer implementing queue import next. I would keep this step as a
read-only ordering preview because it gives us evidence and tests for the next
behavioral boundary without widening authority.
