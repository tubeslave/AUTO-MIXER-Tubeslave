# ADR: Queue Import Memory Trust Review Result Consumer

## Context

The sequence-policy memory queue-import chain can verify memory index results
and produce trust-review candidates. Those candidates are not accepted memory
policy. They require explicit operator acceptance or rejection before any later
promotion planning.

## Options considered

1. Stop after the trust-review request.
2. Add a read-only consumer that verifies trust-review results and emits an
   acceptance-ledger request template.
3. Automatically create accepted records from trust-review candidates.

## Decision

Use option 2. Add a memory-trust-review result consumer and a Product Layer
handler for the existing read-only acceptance ledger builder.

## Why this won

It advances the durable memory workflow while preserving human authority over
trust decisions. The emitted request contains acceptance templates but leaves
`acceptance_records` empty until the operator supplies explicit decisions.

## Rejected alternatives

- Stopping after trust-review leaves the queue-import chain disconnected from
  acceptance evidence.
- Auto-acceptance violates the supervised memory boundary.
- Direct trust-review-to-ledger invocation hides a second action inside a
  verifier.

## Implementation plan

- Verify the memory-index result consumer envelope, summary, and request.
- Verify trust-review envelope, summary, candidates, freshness, and no-mutation
  flags.
- Emit acceptance ledger request templates with no default decision.
- Register the method as `shadow_only` and track it in the project goal matrix.

## Test plan

- Focused unit tests for valid, missing, blocked, and forged evidence.
- Product Layer chain through trust-review result consumer and acceptance ledger
  with empty records.
- Compile touched backend modules, run focused and neighboring pytest suites,
  `tools/memory/memory_lint.py`, `git diff --check`, and full pytest.

## Risks and rollback

Risk: a template could be mistaken for accepted policy. Mitigation: force empty
`acceptance_records`, require explicit operator decisions, and keep acceptance
promotion/execution flags false.

Rollback: remove the consumer, Product Layer handler, registry/matrix entries,
and tests. The previous memory-index-to-trust-review layer remains intact.
