# Brief: Promotion Memory Index Result Consumer

## Task

Add a read-only promotion queue-import consumer that verifies append-only
sequence memory index results against promotion memory-writer apply provenance
before exposing a memory-trust-review request template.

## Definition of Done

- A backend builder verifies promotion writer-apply provenance, generic memory
  index evidence, and fresh sequence-review context.
- Product State dispatch exposes the verifier without invoking trust review,
  mutating memory, policy, runtime state, models, queues, or live mixer state.
- Processing registry and project goal matrix list the method as non-live
  decision evidence.
- Focused tests and ProductState coverage prove the trust-review request
  handoff and rejection paths.
- Memory lint, focused pytest, full pytest, and diff check pass.
