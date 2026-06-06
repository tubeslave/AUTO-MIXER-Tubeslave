# Brief: source-identity-calibration-memory-index-20260606

## Context

Source identity calibration memory drafts can now be written through an
operator-acknowledged append-only writer. The next safe step is to index those
written records as durable, read-only evidence so future replay and trust work
can see that memory exists without changing models or downstream priority.

## Objective

Add a transport-free source identity calibration memory index that parses the
append-only memory blocks emitted by the writer and exposes them in Product
Layer preview as review evidence only.

## Safety Constraints

- Do not mutate source identity models or runtime trust state.
- Do not change downstream proposal priority.
- Do not import indexed records into the decision queue.
- Do not create mixer write requests, `requested_change`, or live apply paths.
- Treat malformed memory blocks, unsafe runtime-authority text, invalid
  timestamps, or unapproved memory paths as invalid.
- Preserve `queue_mutation=false`, `model_mutation=false`,
  `runtime_memory_mutation=false`, `downstream_priority_mutation=false`, and
  `live_write=false`.

## Expected Evidence

- A backend index parses valid append-only writer blocks and fails closed on
  malformed or unsafe records.
- Product Layer preview exposes index status and counts without queue mutation.
- The processing registry and project goal matrix register the index as
  `shadow_only` memory-learning evidence, not a proposal or live executor.
- Focused index, Product Layer, registry, and capability tests pass.
