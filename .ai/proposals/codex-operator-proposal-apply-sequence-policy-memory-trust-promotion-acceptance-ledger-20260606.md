# Proposal: operator-proposal-apply-sequence-policy-memory-trust-promotion-acceptance-ledger-20260606

## Short thesis

Record operator decisions on sequence-memory trust promotion contract records as
evidence, not as sequence-policy or execution authority.

## What I understood

The project is building a durable learning path from successful supervised
apply sequences toward future policy improvement. Contract records are
review-ready evidence, but the operator decision still needs its own auditable
ledger before any later executor planning.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py`.
It consumes the promotion contract payload and caller-supplied acceptance
records, validates record links and safety flags, and emits a read-only ledger
with accepted/rejected records. Execution, queue, policy, model, runtime-memory,
and live-authority flags remain false.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-promotion-acceptance-ledger-20260606.md`

## Alternatives considered

- Treat every contract record as accepted. Rejected: operator review must be
  explicit.
- Let accepted contract records mutate policy immediately. Rejected: acceptance
  is evidence for a later executor, not policy authority.

## Risks

- A forged acceptance record could claim execution or policy authority. The
  ledger rejects unsafe flags and returns replacement-required status.
- A stale contract link could be accepted. The ledger requires exact current
  contract field matching.

## Test plan

- Focused ledger tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may suggest jumping directly to an executor plan. The safer order is to
first land accepted/rejected contract evidence so any executor has a strict,
operator-reviewed input.
