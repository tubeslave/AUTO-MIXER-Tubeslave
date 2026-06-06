# ADR: Promotion Execution-Plan Result Consumer

## Context

The sequence-policy memory trust promotion chain now reaches explicit operator
acceptance and can produce a read-only promotion execution plan. The next step
needs to verify that plan before any dry-run executor receives it.

## Options Considered

1. Let the acceptance-ledger consumer emit dry-run requests directly.
2. Add a separate promotion execution-plan result consumer.
3. Move straight to policy execution.

## Decision

Add a separate promotion execution-plan result consumer that verifies execution
candidates against accepted queue-import promotion records and emits only an
executor dry-run request template.

## Why This Won

The separate consumer keeps the provenance boundary inspectable and prevents an
execution plan from becoming implicit policy authority. It advances the long-term
autonomous learning path while preserving the current fail-closed safety model.

## Rejected Alternatives

- Direct dry-run emission from acceptance-ledger consumption was rejected because
  it skipped a reviewable execution-plan result boundary.
- Direct policy execution was rejected because there is no live or policy-store
  apply authority in this slice.

## Implementation Plan

- Add the consumer module and focused tests.
- Register the method as `decision_evidence`, `shadow_only`.
- Wire Product Layer verifier and separate dry-run handler.
- Add capability matrix and durable memory references.

## Test Plan

- Focused unit tests for transport-free behavior, not-ready state, valid dry-run
  request emission, forged plan rejection, forged prior consumer rejection, and
  registry/matrix coverage.
- Product Layer e2e extension through dry-run without additional manual console
  writes.
- py_compile, focused pytest, memory lint, and diff whitespace checks.

## Risks And Rollback

Risk is accidental authority widening through handler naming or request fields.
Rollback is to remove the verifier registration, Product Layer dispatch entries,
and matrix/docs references. No mixer, queue, policy, model, runtime memory, or
append-only memory writes are introduced.
