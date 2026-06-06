# ADR: Queue Import Memory Trust Acceptance Ledger Result Consumer

## Context

The sequence-policy queue-import memory chain can verify trust-review
candidates and expose acceptance-ledger templates. Those templates deliberately
contain no decisions. A later operator-supplied ledger is required before the
existing promotion-plan builder can produce manual promotion candidates.

## Options considered

1. Stop after the acceptance-ledger request template.
2. Add a read-only consumer that verifies operator-supplied acceptance ledgers
   and emits a promotion-plan request template.
3. Automatically convert trust-review templates into accepted ledger records.
4. Build the promotion plan directly inside the result consumer.

## Decision

Use option 2. Add a memory-trust-acceptance-ledger result consumer and a
Product Layer handler for the existing read-only promotion-plan builder.

## Why this won

It advances the durable memory workflow while preserving explicit operator
authority over trust decisions and keeping promotion planning as a separate,
auditable action. Empty or partially reviewed ledgers remain blocked.

## Rejected alternatives

- Stopping after acceptance leaves the queue-import chain disconnected from
  promotion planning.
- Auto-acceptance violates the supervised memory boundary.
- Building a promotion plan inside the verifier hides an action inside a
  result-consumer step and makes audit trails less clear.

## Implementation plan

- Verify the prior memory-trust-review result consumer envelope, summary,
  request, and templates.
- Verify the memory trust acceptance ledger envelope, summary, decisions,
  record-to-template identity, timestamps, and no-mutation flags.
- Block empty or unreviewed acceptance ledgers.
- Emit only a read-only promotion-plan request template.
- Register the method as `shadow_only` and track it in the project goal matrix.

## Test plan

- Focused unit tests for valid, missing, blocked, and forged evidence.
- Product Layer chain through empty acceptance, explicit acceptance, consumer,
  and promotion-plan request.
- Compile touched backend modules, run focused and neighboring pytest suites,
  `tools/memory/memory_lint.py`, `git diff --check`, and full pytest.

## Risks and rollback

Risk: a valid ledger could be mistaken for promotion authority. Mitigation:
the consumer only emits a promotion-plan request template, keeps all mutation
flags false, and requires the existing promotion plan plus later contract gates.

Rollback: remove the consumer, Product Layer handlers, registry/matrix entries,
and tests. The previous trust-review-to-acceptance-ledger layer remains intact.
