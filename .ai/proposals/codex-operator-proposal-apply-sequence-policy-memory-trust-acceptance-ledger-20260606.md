# Proposal: operator-proposal-apply-sequence-policy-memory-trust-acceptance-ledger-20260606

## Short thesis

Record operator accepted/rejected sequence-memory trust candidates as evidence,
not as promotion or execution authority.

## What I understood

The project needs long-term learning from successful supervised sequence
executions. The trust-review layer surfaces candidates, but the system still
needs an auditable operator decision before any later promotion contract can
consume those candidates.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py`.
It consumes the sequence memory trust-review payload and caller-supplied
acceptance records, validates candidate links and safety flags, and emits a
read-only ledger with accepted/rejected records. All policy, execution, queue,
model, runtime-memory, and live-authority flags remain false.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-trust-acceptance-ledger-20260606.md`

## Alternatives considered

- Let trust-review candidates automatically count as accepted. Rejected:
  operator review must be explicit.
- Let accepted records feed policy immediately. Rejected: acceptance is evidence
  for a future promotion contract, not policy authority.

## Risks

- A forged acceptance record could claim execution or policy authority. The
  ledger rejects unsafe flags and returns replacement-required status.
- A stale candidate link could be accepted. The ledger requires exact candidate
  field matching.

## Test plan

- Focused ledger tests.
- Product Layer preview test.
- Registry/capability matrix tests.
- Compile, memory lint, diff check, and full backend suite.

## Where the other agent may disagree

Kimi may suggest implementing the promotion contract immediately. The safer
sequence is to first land the acceptance ledger so any promotion contract has a
strict, operator-reviewed input surface.
