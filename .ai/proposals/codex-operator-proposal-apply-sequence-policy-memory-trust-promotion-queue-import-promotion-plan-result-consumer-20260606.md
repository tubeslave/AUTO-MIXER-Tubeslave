# Codex Proposal: Queue Import Promotion Plan Result Consumer

## Short thesis

Add a read-only verifier that connects verified sequence-memory promotion plans
to the existing promotion-contract builder without creating contracts or
granting policy authority.

## What I understood

The queue-import chain can now reach a manual promotion plan from explicit
operator acceptance records. A promotion plan is still not a contract, not a
policy mutation, and not execution authority. The safe next transition is to
verify the plan against the previous queue-import acceptance evidence and emit a
contract request that still requires fresh context and rollback evidence.

## Proposed solution

Create a promotion-plan result consumer that accepts the memory-trust
acceptance-ledger result consumer and a promotion plan result. It validates the
prior consumer, request, accepted/rejected record counts, promotion plan
envelope, candidate identities, summary counts, and no-mutation flags. If valid,
it emits a `contract_operator_proposal_apply_sequence_policy_memory_trust_promotion`
request template with the promotion plan plus required context/rollback markers.

## Likely files to touch

- New backend consumer module
- Product Layer handler and dispatch table
- Processing registry and project goal matrix
- Focused unit tests and Product Layer chain test
- `.ai` brief/proposal and ADR
- `.ai/memory/project.md`

## Alternatives considered

- Build the contract inside the verifier: rejected because contract creation is
  a separate action with extra evidence requirements.
- Skip this verifier and feed promotion plan directly into contract: rejected
  because queue-import provenance should be checked at every transition.
- Treat promotion-plan candidates as policy trust: rejected because contract,
  acceptance, execution, audit, and rollback gates still follow.

## Risks

- A promotion plan could be mistaken for policy trust.
- A forged plan could carry unsafe policy/runtime/live flags.
- Missing rollback evidence could be hidden by a prefilled request.

## Test plan

Unit tests should cover missing inputs, valid contract request generation,
not-ready plan blocking, forged plan rejection, forged prior-consumer rejection,
registry exposure, and matrix exposure. Product Layer tests should prove that
the contract is built only after a separate handler call with explicit fresh
context and rollback evidence.

## Where the other agent may disagree

Kimi may argue for fewer micro-consumers. The counterpoint is auditability:
every transition from memory to trust to promotion to contract has separate
evidence, failure modes, and rollback boundaries.
