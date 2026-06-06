# Codex Proposal: Sequence Policy Memory Trust Promotion Policy Consumer

## Short thesis

Add a read-only policy consumer after the sequence-policy gate so eligible
sequence-memory trust evidence can inform future ordering without becoming
ordering, queue, execution, or live authority.

## What I understood

The sequence-memory trust chain now separates acceptance, promotion, executor
dry-run, operator-gated apply, audit, rollback verification, and eligibility.
The next safe step is a consumer that projects eligible gate records onto
current proposal payloads while keeping all mutations blocked.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer.py`.
It validates the gate envelope, summary counters, record identity, duplicate
keys, and safety flags. It matches eligible gate records to preview proposals by
proposal ID, idempotency key, policy key, learning scope, value type, or action
ID, then emits `consumer_records` as read-only policy evidence.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-policy-consumer-20260606.md`

## Alternatives considered

- Let the sequence-policy gate feed a future ordering layer directly. Rejected
  because matching gate evidence to live preview proposals is a separate
  validation boundary.
- Reorder proposals now. Rejected because proposal ordering and queue import
  require separate reviewed contracts.

## Risks

- Consumer records could be mistaken for sequence execution authority.
  Mitigation: keep `proposal_order_mutation=false`, `queue_mutation=false`,
  `sequence_execution_allowed_by_consumer=false`, `sequence_policy_change_allowed_by_consumer=false`,
  `can_import_to_queue=false`, and `can_apply_live=false`.
- Forged gate evidence could claim runtime authority. Mitigation: reject
  forged envelope, summary, and record flags before emitting consumer records.

## Test plan

- Unit tests for not-ready, matched records, forged authority, duplicate policy
  keys, registry, and capability matrix.
- Extend Product Layer preview test for payload, adapter, and summary exposure.
- Run focused pytest and memory lint.

## Where the other agent may disagree

Kimi may prefer adding ordering behavior now. I would keep this as a projection
layer because policy evidence should be reviewed before it can affect order,
queue import, or supervised execution.
