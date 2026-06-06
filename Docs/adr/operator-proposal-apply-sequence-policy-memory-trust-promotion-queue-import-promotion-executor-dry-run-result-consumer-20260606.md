# ADR: Promotion Executor Dry-Run Result Consumer

## Context

Sequence-policy memory trust promotion now has a read-only path from queue-import
promotion records to executor dry-run requests. The dry-run output still needs a
provenance boundary before any executor preview or future apply request.

## Options Considered

1. Let the execution-plan result consumer directly invoke the executor dry-run
   and executor.
2. Add a separate executor dry-run result consumer.
3. Allow direct policy-store mutation after dry-run.

## Decision

Add a separate executor dry-run result consumer that verifies dry-run specs
against queue-import execution-plan provenance and emits only an executor request
template with `apply=False` and `operator_ack=False`.

## Why This Won

This keeps dry-run evidence reviewable and prevents dry-run specs from becoming
implicit sequence-policy mutation authority. It advances the autonomous learning
path while preserving an explicit operator-gated policy-store apply boundary.

## Rejected Alternatives

- Direct executor invocation was rejected because it merges verification and
  execution.
- Direct policy-store mutation was rejected because target store, operator ack,
  and apply intent must remain explicit in a separate executor call.

## Implementation Plan

- Add the consumer module and focused tests.
- Wire Product Layer verifier and executor handler.
- Register the method as `decision_evidence`, `shadow_only`.
- Update the capability matrix, durable project memory, and this ADR.

## Test Plan

- Focused tests for transport-free behavior, not-ready state, request emission,
  forged dry-run rejection, forged prior consumer rejection, and registry/matrix
  coverage.
- Product Layer flow through verifier and executor preview.
- py_compile, focused pytest, memory lint, diff whitespace checks, and backend
  suite.

## Risks And Rollback

Risk is authority widening through request fields. Rollback is to remove the
consumer, Product Layer dispatch entries, registry/matrix references, and docs.
No mixer, queue, policy, model, runtime memory, append-only memory, or
idempotency writes are introduced by the consumer.
