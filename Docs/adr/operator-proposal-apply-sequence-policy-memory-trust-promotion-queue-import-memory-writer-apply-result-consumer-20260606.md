# ADR: Queue Import Memory Writer Apply Result Consumer

## Context

The supervised sequence-policy queue-import chain can verify memory-writer
dry-run evidence and expose a generic writer apply request template. The generic
writer is the only layer that may append to `.ai/memory/project.md`, and only
with explicit operator acknowledgement.

## Options considered

1. Stop after the writer dry-run request template.
2. Add a consumer that verifies applied writer results and emits an index request
   template.
3. Fold indexing directly into the writer result consumer.

## Decision

Use option 2. Add a read-only memory-writer apply result consumer, plus a
Product Layer handler for the existing read-only sequence memory index builder.

## Why this won

It advances long-term memory learning without widening live control. The writer
result can prove append-only file mutation happened, while the consumer remains a
verifier and cannot mutate memory, policy, runtime state, models, queues, or
mixer transport.

## Rejected alternatives

- Stopping at dry-run leaves no queue-import proof path from explicit memory
  application to index review.
- Folding index execution into the consumer makes a verifier perform a second
  operation and weakens auditability.
- Treating applied memory as trusted policy skips the separate memory-index and
  trust-promotion gates.

## Implementation plan

- Verify the dry-run result consumer envelope, summary, and apply request.
- Verify the generic writer applied status, operator acknowledgement, target
  path, draft IDs, write results, and non-live false flags.
- Emit `index_operator_proposal_apply_sequence_policy_memory` as a template only.
- Register the method as `shadow_only` and track it in the project capability
  matrix.
- Extend the Product Layer chain through temp-root writer apply and memory
  indexing.

## Test plan

- Unit tests for not-ready, valid applied writer result, dry-run rejection,
  forged applied result rejection, registry, and capability matrix.
- Product Layer dispatch test through dry run, explicit writer apply, result
  consumer, and read-only memory index.
- Compile touched backend modules, run focused and neighboring pytest suites,
  `tools/memory/memory_lint.py`, `git diff --check`, and full pytest before
  merge.

## Risks and rollback

Risk: a forged writer result could imply memory was persisted. Mitigation:
require matching draft IDs, `write_results`, positive bytes, target path, and
applied writer flags, then let the index builder read the actual file.

Rollback: remove the new consumer method, Product Layer handlers, registry and
matrix entries, and tests. The prior dry-run request-template chain remains
usable.
