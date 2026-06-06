# ADR: Queue Import Promotion Ordering-Preview Result Consumer

## Context

The sequence-policy promotion chain now reaches a verified policy-consumer
result consumer that can request a read-only ordering preview. The repo also has
a read-only queue-ordering-plan builder. A provenance verifier is needed between
those steps so queue planning is not exposed from forged or stale ordering
preview evidence.

## Options Considered

1. Wire queue-ordering planning directly after policy-consumer verification.
2. Accept raw ordering previews at Product Layer without a consumer.
3. Add a fail-closed ordering-preview result consumer.

## Decision

Use option 3. Add a read-only ordering-preview result consumer that validates
prior policy-consumer provenance and ordering-preview rows before emitting a
queue-ordering-plan request template.

## Why This Won

It advances the autonomous sequencing architecture while preserving the
project's current safety boundary. The new step proves ordering evidence before
queue planning and keeps queue import, proposal order mutation, policy mutation,
memory mutation, idempotency reservation, live authority, and Full Autonomous
readiness false.

## Rejected Alternatives

- Direct planning was rejected because it weakens the evidence chain.
- Raw Product Layer preview acceptance was rejected because forged rows could
  carry queue or live authority fields.

## Implementation Plan

- Add the verifier backend module.
- Add Product Layer verifier and direct read-only queue-ordering-plan handler.
- Register the method as `shadow_only` decision evidence.
- Add capability matrix support paths.
- Extend focused and Product Layer tests.
- Record the durable rule in `.ai/memory/project.md`.

## Test Plan

- Focused unit tests for not-ready, blocked, valid, forged prior, and forged
  ordering-preview states.
- Registry and capability matrix non-live assertions.
- Product Layer e2e continuation through verified queue-ordering-plan request.
- Memory lint and broader pytest before merge when feasible.

## Risks And Rollback

The main risk is over-constraining valid future ordering-preview payloads. Roll
back by removing the new consumer, Product Layer handlers, registry/matrix rows,
and tests; no live or queue mutation path is introduced by this ADR.
