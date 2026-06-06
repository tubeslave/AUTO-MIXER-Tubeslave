# Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Promotion Memory Draft Result Consumer

## Context

The promotion audit-policy-gate result consumer can verify generic policy evidence and emit a memory-draft request template. The next safe boundary is to consume a completed generic policy-memory draft, verify it against promotion audit-policy-gate provenance, and expose only a memory-writer request template.

## Scope

- Add a read-only promotion memory-draft result consumer.
- Reuse the generic queue-import memory-draft result consumer for draft validation.
- Register the method in Product State dispatch, the processing registry, and the project-goal capability matrix.
- Add focused tests and Product State dispatch coverage.

## Safety Constraints

- Do not invoke memory writer handlers.
- Do not write `.ai/memory/project.md`.
- Do not mutate policy stores, runtime memory, models, queues, proposal state, or sequence-policy trust stores.
- Do not reserve idempotency keys or call live mixer transport.
- Keep emitted writer templates at `apply=false` and `operator_ack=false`.
- Do not grant live authority or Full Autonomous readiness.
