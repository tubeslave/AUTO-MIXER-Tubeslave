# ADR: Promotion Memory Trust Review Result Consumer

## Context

The promotion queue-import memory chain can now verify a read-only memory-index
result and expose a memory-trust-review request template. The next handoff must
verify the trust-review result before any acceptance-ledger request is exposed.

## Options considered

1. Call the generic memory trust-review result consumer directly.
2. Add a promotion wrapper around the generic memory trust-review result
   consumer.
3. Skip to acceptance-ledger result validation.

## Decision

Add a promotion memory-trust-review result consumer that validates promotion
memory-index provenance, delegates generic trust-review validation, and emits a
promotion-scoped acceptance-ledger request template.

## Why this won

It preserves promotion evidence through the trust-review step while reusing the
generic trust-review result contract. The consumer does not invoke the
acceptance ledger, create acceptance records, mutate memory, mutate policy or
queue state, reserve idempotency, alter proposal state, mutate sequence-policy
trust stores, or touch live mixer transport.

## Rejected alternatives

Direct generic use was rejected because it loses promotion-specific provenance.
Skipping to acceptance-ledger result validation was rejected because acceptance
inputs need a separate verified handoff.

## Implementation plan

- Add a transport-free backend builder.
- Wire Product State dispatch aliases.
- Register the method as `shadow_only` decision evidence.
- Add project goal matrix paths.
- Add focused tests and ProductState e2e assertions.

## Test plan

Run py_compile, focused pytest, full pytest, memory lint, and diff check.

## Risks and rollback

The main risk is confusing trust-review candidates with operator acceptance.
The rollback is to remove the new method, dispatch key, registry entry, matrix
paths, and tests; no live write path or runtime migration is introduced.
