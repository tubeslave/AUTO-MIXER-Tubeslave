# Codex Proposal: Queue Import Memory Trust Acceptance Ledger Result Consumer

## Short thesis

Add a read-only verifier that connects explicit operator memory-trust
acceptance decisions to the existing promotion-plan builder without promoting
policy or executing anything.

## What I understood

The queue-import memory path can now produce trust-review candidates and an
acceptance-ledger request template. The missing link is a verifier for the
result after an operator supplies accepted/rejected records. That verifier must
distinguish "operator has not reviewed yet" from "operator supplied valid
records" and must not infer acceptance itself.

## Proposed solution

Create a memory-trust-acceptance-ledger result consumer that accepts the prior
trust-review result consumer and a memory trust acceptance ledger. It validates
the prior consumer envelope, acceptance templates, ledger envelope, summary
counts, record-to-template identity, operator decisions, timestamps, and all
no-mutation flags. If valid and fully reviewed, it emits a read-only
`plan_operator_proposal_apply_sequence_policy_memory_trust_promotion` request.

## Likely files to touch

- New backend consumer module
- Product Layer handler and dispatch table
- Processing registry and project goal matrix
- Focused unit tests and Product Layer chain test
- `.ai` brief/proposal and ADR
- `.ai/memory/project.md`

## Alternatives considered

- Auto-fill acceptance records from templates: rejected because trust decisions
  must be explicit operator input.
- Let the acceptance ledger feed promotion plan directly: rejected because
  queue-import provenance should be rechecked at the transition.
- Build the promotion plan inside the verifier: rejected because that hides a
  second action inside a result consumer.

## Risks

- Treating a valid acceptance ledger as policy promotion authority.
- Letting a forged ledger carry unsafe policy/runtime/live flags.
- Failing to block partially reviewed candidate sets.

## Test plan

Unit tests should cover missing inputs, valid accepted/rejected records, empty
ledger blocking, forged ledger rejection, forged prior-consumer rejection,
registry exposure, and matrix exposure. Product Layer tests should prove that
empty acceptance remains a stop, while explicit operator records reach only a
read-only promotion-plan request and plan.

## Where the other agent may disagree

Kimi may prefer combining this with the promotion-plan handler to reduce steps.
Keeping it separate makes the safety boundary auditable: verification, planning,
contracting, and future execution remain distinct actions.
