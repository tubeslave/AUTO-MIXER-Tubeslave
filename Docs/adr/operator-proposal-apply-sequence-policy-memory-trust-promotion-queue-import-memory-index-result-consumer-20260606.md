# ADR: Queue Import Memory Index Result Consumer

## Context

The sequence-policy memory queue-import chain can verify writer apply results
and expose a read-only memory-index request. The memory index parses append-only
records but does not by itself make those records trustworthy for policy.

## Options considered

1. Stop after the memory-index request.
2. Add a read-only consumer that verifies the memory-index result and emits a
   trust-review request template.
3. Let the index builder invoke trust-review directly.

## Decision

Use option 2. Add a memory-index result consumer and a Product Layer trust-review
handler. The consumer requires fresh read-only sequence-review context.

## Why this won

It connects durable append-only memory to operator trust-review while preserving
the safety boundary. The transition remains auditable: writer apply, memory
index, and trust-review request are separate payloads with separate evidence.

## Rejected alternatives

- Stopping after index leaves the queue-import proof chain disconnected from
  trust-review.
- Direct index-to-review invocation hides a second action inside a read-only
  parser.
- Skipping fresh context risks stale memory becoming review evidence without a
  current state check.

## Implementation plan

- Verify writer-apply consumer provenance.
- Verify memory-index envelope, summary, records, target path, readback/proof,
  and non-mutating flags.
- Verify fresh review context is read-only and non-mutating.
- Emit only a trust-review request template.
- Register the method as `shadow_only` and track it in the project goal matrix.

## Test plan

- Focused unit tests for valid, missing, stale/unsafe, and forged cases.
- Product Layer chain through temp-root writer apply, index, index-result
  consumer, and trust-review.
- Compile touched backend modules, run focused and neighboring pytest suites,
  `tools/memory/memory_lint.py`, `git diff --check`, and full pytest.

## Risks and rollback

Risk: a forged or stale index could create trust-review candidates. Mitigation:
require matching queue-import provenance, zero invalid index records, verified
readback/proof fields, and fresh read-only context.

Rollback: remove the consumer, Product Layer handlers, registry/matrix entries,
and tests. The previous writer-apply-to-index layer remains intact.
