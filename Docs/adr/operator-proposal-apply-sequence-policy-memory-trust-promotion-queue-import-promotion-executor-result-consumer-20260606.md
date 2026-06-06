# ADR: Promotion Executor Result Consumer

## Context

Sequence-policy memory trust promotion can now produce and verify dry-run specs
for the operator-gated promotion executor. The next step is verifying executor
output before rollback/audit evidence is built.

## Options Considered

1. Let dry-run result consumption skip directly to apply audit.
2. Add a separate promotion executor result consumer.
3. Treat executor preview output as audit-ready.

## Decision

Add a separate promotion executor result consumer that accepts only applied
executor results and emits a read-only apply-audit request template.

## Why This Won

The separate consumer keeps preview, applied executor output, and rollback audit
as distinct reviewable boundaries. It advances the autonomous learning path
without letting preview evidence become policy-audit authority.

## Rejected Alternatives

- Direct audit from dry-run evidence was rejected because it skips the executor
  result boundary.
- Treating preview output as audit-ready was rejected because no policy-store
  mutation occurred.

## Implementation Plan

- Add the result-consumer module and focused tests.
- Wire Product Layer verifier and apply-audit handler.
- Register the method as `decision_evidence`, `shadow_only`.
- Update capability matrix, durable memory, and this ADR.

## Test Plan

- Focused tests for transport-free behavior, not-ready state, preview blocking,
  applied request emission, forged executor rejection, forged prior consumer
  rejection, and registry/matrix coverage.
- Product Layer flow through preview block, applied executor, verifier, and
  read-only apply audit.
- py_compile, focused pytest, memory lint, diff whitespace checks, promotion
  chain, and backend suite.

## Risks And Rollback

Risk is confusing policy-store apply with live console authority. Rollback is to
remove the consumer, Product Layer dispatch entries, registry/matrix references,
and docs. The consumer itself introduces no mixer, queue, policy, model, runtime
memory, append-only memory, or idempotency writes.
