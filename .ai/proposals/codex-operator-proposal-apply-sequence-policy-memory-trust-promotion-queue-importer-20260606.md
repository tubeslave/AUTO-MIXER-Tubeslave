# Codex Proposal: Sequence Policy Memory Trust Promotion Queue Importer

## Short thesis

Add an operator-gated queue importer after the sequence-policy queue-ordering
plan so trusted sequence-memory ordering evidence can be turned into pending
operator proposals without granting live or direct sequence authority.

## What I understood

The project is moving toward autonomous apply-sequence behavior, but the current
safe boundary is still Assisted Mixing. Queue-ordering plan records should be
importable only through an explicit operator action, never implicitly from the
preview path.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer.py`.
It validates the queue-ordering plan envelope, summary counters, record safety
flags, duplicate identifiers, planned indexes, and proposal payload coverage.
It emits dry-run import specs by default. With `apply=true`,
`operator_ack=true`, and a target `OperatorProposalQueue`, it calls only that
queue's `create` method.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/handlers/product_state_handlers.py`
- `backend/operator_product_state.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-importer-20260606.md`

## Alternatives considered

- Let the Product Layer preview import automatically. Rejected because preview
  must remain non-mutating.
- Feed queue-ordering records directly into sequence execution. Rejected because
  sequence execution needs a separate consumer, preflight evidence, and runtime
  gates.

## Risks

- Queue import could be mistaken for live apply authority. Mitigation: keep
  imported proposals behind the existing proposal apply preflight and preserve
  `can_apply_live=false`.
- Forged plan records could claim queue or live authority. Mitigation: reject
  forged envelope, summary, and record-level authority before producing specs.

## Test plan

- Unit-test not-ready, missing payloads, dry-run, ack/target queue gates,
  explicit queue import, forged authority, registry, and capability matrix.
- Extend Product Layer preview and handler tests.
- Run focused pytest and memory lint.

## Where the other agent may disagree

Kimi may prefer jumping directly to sequence execution. I would keep this layer
queue-only because pending operator proposals still preserve review, preflight,
and safety gates while advancing toward autonomous orchestration.
