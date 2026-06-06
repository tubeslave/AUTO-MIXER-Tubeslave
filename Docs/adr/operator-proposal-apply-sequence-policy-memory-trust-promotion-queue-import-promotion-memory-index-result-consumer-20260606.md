# ADR: Promotion Memory Index Result Consumer

## Context

The promotion queue-import memory chain can now verify an explicitly applied
append-only writer result and expose a memory-index request template. The next
handoff must verify the read-only index result before any trust-review request
is exposed.

## Options considered

1. Call the generic memory-index result consumer directly.
2. Add a promotion wrapper around the generic memory-index result consumer.
3. Skip to trust-review result validation.

## Decision

Add a promotion memory-index result consumer that validates promotion
writer-apply provenance, delegates generic index validation, requires fresh
sequence-review context, and emits a promotion-scoped trust-review request
template.

## Why this won

It preserves promotion evidence through the memory-index step while reusing the
generic index contract. The consumer does not invoke trust review, mutate
memory, mutate policy or queue state, reserve idempotency, or touch live mixer
transport.

## Rejected alternatives

Direct generic use was rejected because it loses promotion-specific provenance.
Skipping to trust-review result validation was rejected because trust-review
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

The main risk is confusing memory index evidence with policy or execution
authority. The rollback is to remove the new method, dispatch key, registry
entry, matrix paths, and tests; no live write path or runtime migration is
introduced.
