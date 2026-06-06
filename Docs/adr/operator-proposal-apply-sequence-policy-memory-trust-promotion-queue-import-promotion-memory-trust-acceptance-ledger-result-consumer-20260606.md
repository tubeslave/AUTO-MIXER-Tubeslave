# ADR: Promotion Memory Trust Acceptance Ledger Result Consumer

## Context

The promotion queue-import memory chain can now verify memory trust-review
results and expose an acceptance-ledger request template. The next handoff must
verify explicit operator accept/reject records before any promotion-plan request
is exposed.

## Options considered

1. Call the generic memory trust acceptance-ledger result consumer directly.
2. Add a promotion wrapper around the generic acceptance-ledger result
   consumer.
3. Skip to promotion-plan result validation.

## Decision

Add a promotion memory-trust-acceptance-ledger result consumer that validates
promotion memory-trust-review provenance, delegates generic acceptance-ledger
validation, and emits a promotion-scoped promotion-plan request template.

## Why this won

It preserves promotion evidence through the acceptance-ledger step while reusing
the generic acceptance-ledger result contract. The consumer does not invoke
promotion planning, promote policy, mutate memory, mutate policy or queue
state, reserve idempotency, alter proposal state, mutate sequence-policy trust
stores, or touch live mixer transport.

## Rejected alternatives

Direct generic use was rejected because it loses promotion-specific provenance.
Skipping to promotion-plan result validation was rejected because promotion-plan
inputs need a separate verified handoff.

## Implementation plan

- Add a transport-free backend builder.
- Wire Product State dispatch aliases.
- Allow the existing promotion-plan result consumer to accept the new
  promotion-scoped prior while preserving the existing generic prior.
- Register the method as `shadow_only` decision evidence.
- Add project goal matrix paths.
- Add focused tests and ProductState e2e assertions.

## Test plan

Run py_compile, focused pytest, full pytest, memory lint, and diff check.

## Risks and rollback

The main risk is confusing operator acceptance records with policy promotion.
The rollback is to remove the new method, dispatch key, registry entry, matrix
paths, compatibility alias, and tests; no live write path or runtime migration
is introduced.
