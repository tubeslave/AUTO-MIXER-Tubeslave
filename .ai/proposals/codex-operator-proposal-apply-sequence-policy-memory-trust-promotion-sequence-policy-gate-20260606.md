# Codex Proposal: Sequence Policy Memory Trust Promotion Policy Gate

## Short thesis

Add a read-only sequence-policy gate after rollback verification so verified
memory trust promotion can become eligibility evidence without becoming runtime
policy authority.

## What I understood

The project is moving toward more autonomous supervised sequences, but the trust
chain must remain explicit. Verified rollback records prove that a promotion can
be reviewed safely; they should not directly mutate queue order, policy stores,
or execution behavior.

## Proposed solution

Create
`backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate.py`.
It validates the rollback-verification envelope, summary counters, verified
record identity, and safety flags. It emits `policy_records` only when the target
policy state is `trusted_for_sequence_policy`.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate.py`
- `tests/test_operator_product_state.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-sequence-policy-gate-20260606.md`

## Alternatives considered

- Let rollback verification feed a future sequence policy consumer directly.
  Rejected because that skips an explicit eligibility decision.
- Add queue ordering or execution changes now. Rejected because policy eligibility
  and runtime behavior should remain separate reviewed contracts.

## Risks

- A policy gate could be mistaken for direct execution authority. Mitigation:
  keep `sequence_execution_allowed_by_gate=false`, `sequence_policy_change_allowed_by_gate=false`,
  `can_import_to_queue=false`, and `can_apply_live=false`.
- Forged verified records could claim live or policy-store mutation. Mitigation:
  reject forged record-level and summary-level authority.

## Test plan

- Unit tests for not-ready, eligible, forged authority, ineligible policy state,
  registry, and capability matrix.
- Extend Product Layer preview test for payload, adapter, and summary exposure.
- Run focused pytest and memory lint.

## Where the other agent may disagree

Kimi may prefer jumping to a queue/order consumer. I would keep this as a narrow
gate because policy eligibility is the prerequisite for any later ordering or
execution change.
