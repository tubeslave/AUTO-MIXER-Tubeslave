# Codex Proposal: Queue Import Memory Index Result Consumer

## Short thesis

Add a read-only verifier that connects verified queue-import memory indexing to
memory trust-review without granting policy or execution authority.

## What I understood

After an operator-approved writer append and read-only index pass, the system
still needs to prove that indexed memory matches the same queue-import
provenance before trust-review candidates are built. Fresh review context is
required so stale memory cannot immediately influence policy.

## Proposed solution

Create a memory-index result consumer that accepts a verified writer-apply
consumer, a memory index result, and fresh read-only review context. It validates
record counts, proposal IDs, draft IDs, target path, index safety flags,
readback/proof evidence, and freshness flags. If valid, it emits only a
`review_operator_proposal_apply_sequence_policy_memory_trust` request template.

## Likely files to touch

- New backend consumer module
- Product Layer handlers for consumer and trust-review
- Processing registry and project goal matrix
- Focused unit tests and Product Layer chain test
- `.ai` brief/proposal and ADR
- `.ai/memory/project.md`

## Alternatives considered

- Let the memory index feed trust-review directly: rejected because queue-import
  provenance should be rechecked at the transition.
- Auto-run trust-review inside the consumer: rejected because the verifier would
  perform a second operation and hide handler boundaries.
- Skip fresh context: rejected because long-term memory should not affect review
  without a current read-only context check.

## Risks

- Treating indexed memory as trusted policy instead of review evidence.
- Accepting a forged index with mismatched proposal or draft IDs.
- Letting stale context create review candidates.

## Test plan

Unit tests should cover not-ready inputs, valid request generation, missing or
unsafe freshness, forged index evidence, registry exposure, and matrix exposure.
Product Layer tests should dispatch through writer apply, index, index-result
consumer, and trust-review using a temp memory root.

## Where the other agent may disagree

Kimi may argue for a separate freshness producer before this consumer. That is a
reasonable future split. This layer keeps freshness explicit in the request and
validates it as read-only evidence before trust-review can be requested.
