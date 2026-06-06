# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-contract-20260606

## Short thesis

Require fresh sequence-policy context and rollback evidence before accepted
sequence-memory trust candidates can be reviewed for promotion.

## What I understood

The project needs long-term memory to influence future supervised sequence
policy eventually, but accepted memory records must not become policy authority
by accident. A contract boundary makes the next step explicit and reviewable.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py`.
It consumes the manual promotion plan, validates the plan envelope and
candidates, checks fresh read-only sequence-policy context, checks rollback
evidence, and emits contract records for operator review. All execution, queue,
policy, model, runtime-memory, and live-authority flags remain false.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-contract-20260606.md`

## Alternatives considered

- Let promotion-plan candidates advance without rollback evidence. Rejected:
  policy influence needs a known recovery path before review.
- Combine contract and executor. Rejected: the contract is evidence only and
  should not mutate policy or execution state.

## Risks

- A forged context payload could claim a live apply happened. The contract
  rejects `live_write`, `apply_attempted`, queue mutation, and policy mutation.
- A forged rollback payload could hide policy mutation. The contract rejects
  unsafe rollback flags and unknown strategies.

## Test plan

- Focused contract tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may push for a policy executor now. The safer next step is a strict
contract with rollback evidence; executor behavior should remain a separate
reviewed boundary.
