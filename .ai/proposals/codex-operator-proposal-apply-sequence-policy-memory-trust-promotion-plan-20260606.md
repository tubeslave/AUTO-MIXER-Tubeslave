# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-plan-20260606

## Short thesis

Surface accepted sequence-memory trust records as manual promotion candidates,
not as policy or execution authority.

## What I understood

The project needs durable learning from supervised fader/gain sequence
executions. Acceptance records are useful evidence, but they still need a
separate manual promotion plan before any later contract can consider policy or
execution influence.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py`.
It consumes the sequence-memory trust acceptance ledger, validates the ledger
envelope, summary counts, accepted records, and safety flags, and emits manual
promotion candidates. Every policy, execution, queue, model, runtime-memory, and
live-authority flag remains false.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-plan-20260606.md`

## Alternatives considered

- Let accepted ledger records directly influence sequence policy. Rejected:
  acceptance is evidence, not promotion authority.
- Skip the plan and jump to an executor. Rejected: the system needs a visible,
  reviewable contract boundary before any policy or execution path.

## Risks

- A forged ledger could claim promotion or execution authority. The plan rejects
  unsafe ledger and record flags and returns replacement-required status.
- Operators could mistake candidates for apply readiness. The payload keeps
  manual-promotion and no-authority flags explicit.

## Test plan

- Focused promotion-plan tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may argue for landing the promotion contract in the same patch. The safer
sequence is to first land a strict plan surface so the later contract has a
small, auditable input.
