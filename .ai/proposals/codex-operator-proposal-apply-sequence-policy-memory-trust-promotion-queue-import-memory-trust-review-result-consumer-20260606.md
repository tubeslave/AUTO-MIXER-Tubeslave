# Codex Proposal: Queue Import Memory Trust Review Result Consumer

## Short thesis

Add a read-only verifier that connects verified memory trust-review candidates to
operator acceptance templates without choosing acceptance decisions.

## What I understood

The queue-import memory chain can now reach trust-review candidates from an
operator-approved append-only memory write. The next safe transition is to
prepare the acceptance ledger surface while preserving the requirement that an
operator explicitly accepts or rejects each candidate.

## Proposed solution

Create a memory-trust-review result consumer that accepts the memory-index result
consumer and a memory trust-review result. It validates candidate counts,
proposal IDs, draft IDs, memory record IDs, value type, verified readback/proof,
freshness, and all no-mutation flags. If valid, it emits only a
`record_operator_proposal_apply_sequence_policy_memory_trust_acceptance` request
template with empty `acceptance_records` and per-candidate templates.

## Likely files to touch

- New backend consumer module
- Product Layer handlers for consumer and acceptance ledger
- Processing registry and project goal matrix
- Focused unit tests and Product Layer chain test
- `.ai` brief/proposal and ADR
- `.ai/memory/project.md`

## Alternatives considered

- Auto-accept candidates: rejected because trust decisions require an operator.
- Let trust-review feed the ledger directly: rejected because queue-import
  provenance should be rechecked at the transition.
- Emit a promotion contract request: rejected because acceptance is only evidence
  and promotion remains a separate gate.

## Risks

- Accidentally encoding default acceptance.
- Treating acceptance as policy promotion.
- Accepting forged trust-review candidates with unsafe authority flags.

## Test plan

Unit tests should cover not-ready inputs, valid request generation, forged
trust-review rejection, not-ready trust-review rejection, registry exposure, and
matrix exposure. Product Layer tests should dispatch through trust-review result
consumer and the acceptance ledger handler with empty records to prove the
operator-review-required boundary.

## Where the other agent may disagree

Kimi may prefer a UI-specific acceptance-template format. This implementation
keeps the template plain and backend-verifiable; a UI layer can later render the
same fields without changing the safety contract.
