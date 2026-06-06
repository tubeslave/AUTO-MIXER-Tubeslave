# Brief: operator-proposal-apply-sequence-policy-memory-trust-promotion-queue-import-memory-draft-result-consumer-20260606

## Context

Sequence-policy queue imports can now be verified through audit-policy gate
results and generic memory-draft request templates. The next provenance gap is
between the generic policy-memory draft result and the append-only memory writer.

## Goal

Add a transport-free consumer that verifies a generic
`operator_proposal_apply_sequence_policy_memory_draft` payload against the
sequence-policy queue-import audit-policy-gate result consumer before emitting a
memory-writer request template.

## Constraints

- Do not call the memory writer, memory draft handler, policy gate, executor,
  apply handler, or live mixer transport.
- Do not write `.ai/memory/project.md`.
- Do not reserve idempotency keys.
- Do not mutate memory stores, queue, proposal state, policy stores, runtime
  memory, or models.
- Keep live authority and full autonomous readiness false.
- Require `memory_draft_ready` records, append-only target
  `.ai/memory/project.md`, fader/gain evidence, verified readback/proof,
  rollback availability, and separate writer acknowledgement.

## Validation

- Focused module tests for ready, blocked, and forged memory-draft evidence.
- Product Layer chain test through memory draft into memory-writer dry-run
  without applying a memory write.
- Registry and capability-matrix assertions.
