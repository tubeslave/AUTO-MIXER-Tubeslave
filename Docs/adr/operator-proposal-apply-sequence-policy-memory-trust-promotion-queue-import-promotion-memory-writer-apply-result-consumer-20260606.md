# ADR: Promotion Memory Writer Apply Result Consumer

## Context

The sequence-policy promotion queue-import path now reaches a promotion
memory-writer dry-run result consumer. The next safe step is to verify the
operator-authorized generic append-only writer result before any memory-index
request is exposed.

## Options considered

1. Call the generic writer-apply result consumer directly.
2. Add a promotion wrapper around the generic writer-apply result consumer.
3. Skip to memory-index result validation.

## Decision

Add a promotion memory-writer apply result consumer that validates promotion
dry-run provenance, delegates generic writer-apply validation, and emits a
promotion-scoped memory-index request template.

## Why this won

It preserves the promotion chain evidence while reusing the generic writer
contract for append-only memory writes. The consumer does not invoke memory
indexing, write files, mutate policy or queue state, reserve idempotency, or
touch live mixer transport.

## Rejected alternatives

Direct generic use was rejected because it loses promotion-specific provenance.
Skipping to memory indexing was rejected because the applied writer result
needs a separate verified handoff.

## Implementation plan

- Add a transport-free backend builder.
- Wire Product State dispatch aliases.
- Register the method as `shadow_only` decision evidence.
- Add project goal matrix paths.
- Add focused tests and ProductState e2e assertions.

## Test plan

Run py_compile, focused pytest, full pytest, memory lint, and diff check.

## Risks and rollback

The main risk is confusing source writer mutation with consumer mutation. The
rollback is to remove the new method, dispatch key, registry entry, matrix
paths, and tests; no live write path or runtime state migration is introduced.
