# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-writer-dry-run-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through memory draft and
generic memory-writer request templates. The next provenance gap is between the
generic memory-writer dry-run result and any future append-only memory write.

## Goal

Add a transport-free consumer that verifies a generic
`operator_proposal_apply_sequence_policy_memory_writer` dry-run payload against
the sequence-policy queue-import memory-draft result consumer before emitting a
memory-writer apply request template.

## Constraints

- Do not call the memory writer, memory draft handler, policy gate, executor,
  apply handler, or live mixer transport.
- Do not write `.ai/memory/project.md`.
- Accept only dry-run writer evidence, not applied writer evidence.
- Do not reserve idempotency keys.
- Keep live authority and full autonomous readiness false.
- Require the expected dry-run blocker:
  `operator_ack_and_apply_required_before_append_only_memory_write`.

## Validation

- Focused module tests for ready dry-run, forged dry-run, and already-applied
  writer evidence.
- Product Layer chain test through writer dry-run into apply request template
  without applying a memory write.
- Registry and capability-matrix assertions.
