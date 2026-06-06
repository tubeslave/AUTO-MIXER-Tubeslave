# Codex Proposal: Sequence Policy Memory Trust Promotion Rollback Verification

## Short thesis

Add a read-only rollback verification gate between sequence-policy promotion
apply-audit records and any future sequence-policy trust consumer.

## What I understood

Applied sequence-policy memory trust promotion is intentionally narrow: it can
mutate only an explicit caller-provided `operator_sequence_policy_trust_gate`
store with `apply=true` and `operator_ack=true`. The apply audit records the
rollback target, but those records should not become trust authority until an
operator verifies rollback evidence.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification.py`.
It validates the apply-audit envelope, record safety flags, summary counts, and
operator verification records. When every apply-audit record has matching
operator-reviewed evidence, it emits `verified_records` with
`rollback_verified=true` and `usable_as_sequence_policy_gate_evidence=true`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-rollback-verification-20260606.md`

## Alternatives considered

- Fold rollback verification into apply audit. Rejected because apply audit and
  operator-reviewed rollback proof are distinct trust boundaries.
- Allow verified records to directly change sequence policy. Rejected because a
  separate gate should decide how verified evidence affects policy consumers.

## Risks

- Forged verification evidence could imply execution authority if flags are not
  validated. Mitigation: reject live, queue, runtime-memory, policy-store,
  proposal-state, and direct sequence-execution authority at envelope and record
  level.
- Preview could imply readiness without applied records. Mitigation: Product
  Layer remains `not_ready` because it calls the executor with `apply=false`.

## Test plan

- Unit tests for not-ready, unverified, verified, forged-authority, registry, and
  capability matrix cases.
- Focused Product Layer preview test for payload/adapter/summary exposure.
- `python3 tools/memory/memory_lint.py`.

## Where the other agent may disagree

Kimi may argue for adding the final sequence-policy gate now. I would keep this
slice narrow: rollback verification is the prerequisite evidence layer, while
policy consumption should remain a separate reviewed contract.
