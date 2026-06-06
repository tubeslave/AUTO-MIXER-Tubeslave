# ADR: Queue Import Promotion Plan Result Consumer

## Context

The sequence-policy queue-import memory chain can verify operator acceptance
ledgers and request a promotion plan. The promotion plan surfaces manual
candidates only; it does not mutate policy, execute proposals, or grant live
authority. A contract requires fresh sequence-policy context and rollback
evidence.

## Options considered

1. Stop after the promotion-plan request.
2. Add a read-only result consumer that verifies promotion-plan results and
   emits a promotion-contract request template.
3. Build the promotion contract directly inside the result consumer.
4. Treat promotion-plan candidates as policy trust.

## Decision

Use option 2. Add a promotion-plan result consumer and Product Layer contract
handler wiring for the existing read-only contract builder.

## Why this won

It advances the memory-to-policy trust workflow while keeping contract creation
and all later promotion/execution steps separate. The request explicitly
requires fresh sequence-policy context and rollback evidence.

## Rejected alternatives

- Stopping after promotion plan leaves the queue-import chain disconnected from
  contract review.
- Building the contract inside the verifier hides a second action and extra
  evidence requirements.
- Treating a plan as policy trust violates the supervised promotion ladder.

## Implementation plan

- Verify the prior acceptance-ledger result consumer envelope, summary, request,
  and no-mutation flags.
- Verify the promotion plan envelope, summary, candidates, accepted/rejected
  record identity, timestamps, and no-mutation flags.
- Emit only a promotion-contract request template requiring context and rollback
  evidence.
- Register the method as `shadow_only` and track it in the project goal matrix.

## Test plan

- Focused unit tests for valid, missing, blocked, and forged evidence.
- Product Layer chain through promotion-plan result consumer and separate
  contract handler call.
- Compile touched backend modules, run focused and neighboring pytest suites,
  `tools/memory/memory_lint.py`, `git diff --check`, and full pytest.

## Risks and rollback

Risk: a valid promotion plan could be mistaken for policy authority. Mitigation:
the consumer emits only a request template, all mutation flags remain false, and
the contract/executor/audit/rollback path stays separate.

Rollback: remove the consumer, Product Layer handlers, registry/matrix entries,
and tests. The previous acceptance-ledger-to-promotion-plan layer remains intact.
