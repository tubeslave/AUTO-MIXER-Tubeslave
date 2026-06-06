# ADR: Queue Import Promotion Acceptance Ledger Result Consumer

## Context

The sequence-policy queue-import memory chain can verify promotion contracts and
request explicit operator acceptance. Accepted promotion records are still only
evidence; execution planning is separate and does not mutate policy or live
state.

## Options considered

1. Stop after promotion acceptance recording.
2. Add a read-only result consumer that verifies promotion acceptance-ledger
   results and emits a promotion execution-plan request template.
3. Build the execution plan directly inside the result consumer.
4. Treat accepted promotion records as sequence-policy trust.

## Decision

Use option 2. Add a promotion acceptance-ledger result consumer and Product
Layer execution-plan handler wiring for the existing read-only execution-plan
builder.

## Why this won

It advances the memory-to-policy trust workflow while preserving the boundary
between accepted evidence, execution planning, and actual policy mutation. The
consumer requires at least one accepted promotion record and still emits only a
template.

## Rejected alternatives

- Stopping after acceptance leaves the queue-import chain disconnected from the
  existing execution-plan layer.
- Building the plan inside the verifier hides a separate action.
- Treating acceptance as policy trust violates the supervised promotion ladder.

## Implementation plan

- Verify the prior promotion-contract result consumer envelope, summary,
  request, and no-mutation flags.
- Verify promotion acceptance ledger envelope, status, summary counts,
  accepted/rejected record identity, timestamps, and no-mutation flags.
- Emit only a promotion execution-plan request template.
- Register the method as `shadow_only` and track it in the project goal matrix.

## Test plan

- Focused unit tests for valid, missing, rejected-only, forged-ledger, and
  forged-prior-consumer evidence.
- Product Layer chain through promotion acceptance-ledger result consumer and
  separate execution-plan handler call.
- Compile touched backend modules, run focused and neighboring pytest suites,
  `tools/memory/memory_lint.py`, and `git diff --check`.

## Risks and rollback

Risk: accepted promotion evidence could be mistaken for policy execution.
Mitigation: the consumer emits only a request template and keeps execution,
policy, queue, memory, model, live authority, and autonomous readiness flags
false.

Rollback: remove the consumer, Product Layer handlers, registry/matrix entries,
and tests. The previous promotion-contract-to-acceptance-ledger layer remains
intact.
