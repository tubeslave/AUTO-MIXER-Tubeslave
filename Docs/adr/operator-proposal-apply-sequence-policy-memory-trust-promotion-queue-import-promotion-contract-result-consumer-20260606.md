# ADR: Queue Import Promotion Contract Result Consumer

## Context

The sequence-policy queue-import memory chain can verify promotion-plan results
and request a read-only promotion contract. A ready promotion contract is still
not policy authority: operator accept/reject records must be supplied before any
later promotion path can continue.

## Options considered

1. Stop after promotion-contract creation.
2. Add a read-only result consumer that verifies promotion-contract results and
   emits a promotion acceptance-ledger request template.
3. Build the promotion acceptance ledger directly inside the result consumer.
4. Treat a ready contract as accepted sequence-policy trust.

## Decision

Use option 2. Add a promotion-contract result consumer and Product Layer
acceptance-ledger handler wiring for the existing read-only acceptance builder.

## Why this won

It advances the memory-to-policy trust workflow while preserving the explicit
operator decision boundary. The emitted request keeps `acceptance_records`
empty, so no contract is accepted by the verifier.

## Rejected alternatives

- Stopping after the contract leaves the queue-import chain disconnected from
  the existing acceptance ledger.
- Building the ledger inside the verifier hides a separate operator action.
- Treating a ready contract as trust violates the supervised promotion ladder.

## Implementation plan

- Verify the prior promotion-plan result consumer envelope, summary, request,
  and no-mutation flags.
- Verify the promotion contract envelope, ready status, summary counts, record
  identity, rollback fields, timestamps, and no-mutation flags.
- Emit only a promotion acceptance-ledger request template with empty
  `acceptance_records`.
- Register the method as `shadow_only` and track it in the project goal matrix.

## Test plan

- Focused unit tests for valid, missing, not-ready, forged-contract, and
  forged-prior-consumer evidence.
- Product Layer chain through promotion-contract result consumer and separate
  acceptance-ledger handler call with empty records.
- Compile touched backend modules, run focused and neighboring pytest suites,
  `tools/memory/memory_lint.py`, `git diff --check`, and full pytest.

## Risks and rollback

Risk: a valid promotion contract could be mistaken for accepted policy trust.
Mitigation: the consumer emits only a request template, leaves acceptance records
empty, and keeps all mutation, execution, live authority, and autonomous
readiness flags false.

Rollback: remove the consumer, Product Layer handlers, registry/matrix entries,
and tests. The previous promotion-plan-to-contract layer remains intact.
