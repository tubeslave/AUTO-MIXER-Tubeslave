# ADR: Queue Import Promotion Queue-Ordering-Plan Result Consumer

## Context

The sequence-policy promotion chain can now verify ordering-preview results and
request read-only queue-ordering planning. The queue importer exists and can
mutate an explicit target queue only with `apply=true` and `operator_ack=true`.
A provenance verifier is needed before exposing importer requests from
queue-ordering-plan output.

## Options Considered

1. Call the queue importer directly after queue-ordering planning.
2. Accept raw queue-ordering plans at Product Layer without a consumer.
3. Add a fail-closed queue-ordering-plan result consumer.

## Decision

Use option 3. Add a read-only queue-ordering-plan result consumer that validates
prior ordering-preview provenance and queue-ordering-plan records before
emitting a queue-importer request template.

## Why This Won

It advances the autonomous queue-import architecture while preserving the
current safety boundary. The new step proves queue-ordering evidence before
importer use and keeps queue import, proposal order mutation, policy mutation,
memory mutation, idempotency reservation, live authority, and Full Autonomous
readiness false.

## Rejected Alternatives

- Direct importer invocation was rejected because queue mutation is a separate
  operator-gated boundary.
- Raw Product Layer plan acceptance was rejected because forged records could
  carry queue or live authority fields.

## Implementation Plan

- Add the verifier backend module.
- Add Product Layer verifier wiring.
- Register the method as `shadow_only` decision evidence.
- Add capability matrix support paths.
- Extend focused and Product Layer tests.
- Record the durable rule in `.ai/memory/project.md`.

## Test Plan

- Focused unit tests for not-ready, blocked, valid, forged prior, and forged
  queue-ordering-plan states.
- Registry and capability matrix non-live assertions.
- Product Layer e2e continuation through verified queue-importer dry-run
  request.
- Memory lint and broader pytest before merge when feasible.

## Risks And Rollback

The main risk is over-constraining valid future queue-ordering-plan payloads.
Roll back by removing the new consumer, Product Layer handler, registry/matrix
rows, and tests; no live or queue mutation path is introduced by this ADR.
