# Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Promotion Memory Writer Dry Run Result Consumer

## Context

The promotion memory-draft result consumer can verify generic policy-memory drafts and emit a memory-writer request template. The next safe boundary is to consume a generic memory-writer dry-run result, verify it against promotion memory-draft provenance, and expose only a memory-writer apply request template.

## Scope

- Add a read-only promotion memory-writer dry-run result consumer.
- Reuse the generic queue-import memory-writer dry-run result consumer for dry-run validation.
- Register the method in Product State dispatch, the processing registry, and the project-goal capability matrix.
- Add focused tests and Product State dispatch coverage.

## Safety Constraints

- Do not invoke memory writer handlers.
- Do not write `.ai/memory/project.md`.
- Do not mutate policy stores, runtime memory, models, queues, proposal state, or sequence-policy trust stores.
- Do not reserve idempotency keys or call live mixer transport.
- Keep emitted apply templates at `apply=false` and `operator_ack=false`.
- Do not grant live authority or Full Autonomous readiness.
