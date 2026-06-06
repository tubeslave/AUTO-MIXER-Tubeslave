# Brief: Promotion Memory Writer Apply Result Consumer

## Task

Add a read-only promotion queue-import consumer that verifies explicitly
applied generic append-only memory-writer results against promotion
memory-writer dry-run provenance before exposing a memory-index request
template.

## Definition of Done

- A backend builder verifies promotion dry-run provenance and generic writer
  apply evidence.
- Product State dispatch exposes the verifier without invoking memory index,
  memory writes, queue mutation, policy mutation, or live mixer writes.
- Processing registry and project goal matrix list the method as non-live
  decision evidence.
- Focused tests and ProductState coverage prove the memory-index handoff and
  rejection paths.
- Memory lint, focused pytest, full pytest, and diff check pass.
