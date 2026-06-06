# Proposal: operator-proposal-apply-sequence-policy-memory-trust-review-20260606

## Short thesis

Turn indexed supervised sequence memory into operator-review candidates, not
execution authority.

## What I understood

The project needs long-term learning from verified supervised actions. The
sequence memory index gives durable evidence, but it should not influence policy
or execution until a separate review and promotion chain exists.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_review.py`.
It consumes the sequence memory index, verifies the index envelope, summary, and
record-level safety flags, checks fresh read-only review context, and emits
operator-review candidates with all execution, queue, policy, model,
runtime-memory, and live-authority flags false.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_review.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_review.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-review-20260606.md`

## Alternatives considered

- Promote indexed memory directly into policy. Rejected: review evidence is not
  policy authority.
- Skip freshness context. Rejected: stale context would make memory look more
  trustworthy than current evidence supports.

## Risks

- Forged memory index payloads could claim policy or live authority. The review
  layer validates the envelope, summary, records, and freshness context and
  returns replacement-required with zero candidates when inconsistent.

## Test plan

- Focused trust-review tests.
- Product Layer preview visibility test.
- Registry and capability matrix checks.
- Compile, memory lint, and full backend suite when the slice stabilizes.

## Where the other agent may disagree

Kimi may prefer designing the full sequence policy promotion chain now. The
smaller review layer is safer because it establishes a narrow, testable boundary
before any accepted promotion or policy-consumer work.
