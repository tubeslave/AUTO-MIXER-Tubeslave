# Codex Proposal: Queue Import Memory Writer Apply Result Consumer

## Short thesis

Add a read-only verifier for explicitly applied generic memory-writer results and
emit only a memory-index request template.

## What I understood

The previous queue-import layer proves a generic memory writer dry run and emits
an apply request that still requires an operator to set `apply=True` and
`operator_ack=True`. After that separate writer actually appends memory, the
chain needs evidence that the applied result matches the verified dry-run
provenance before the memory index is allowed to read the append-only file.

## Proposed solution

Create a transport-free consumer that accepts the dry-run result consumer plus a
generic memory-writer apply result. It should require
`append_only_memory_write_applied`, `success=True`, explicit operator
acknowledgement, matching draft IDs, positive `bytes_appended`, target path
`.ai/memory/project.md`, and no live/policy/runtime/model/queue mutation flags.
The output exposes a template for `index_operator_proposal_apply_sequence_policy_memory`
and records that any source memory mutation came from the writer, not this
consumer.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- Product Layer and focused unit tests
- `.ai/memory/project.md`
- ADR/brief/proposal artifacts

## Alternatives considered

- Index automatically inside the consumer: rejected because that would hide a
  second handler action inside a verifier.
- Treat writer apply as direct policy promotion: rejected because indexed memory
  is review evidence only and still needs separate trust/policy promotion.
- Accept any writer result status: rejected because only explicit applied writer
  payloads prove append-only persistence.

## Risks

- Confusing source memory-file mutation with consumer-side mutation.
- Accepting forged write results with wrong draft IDs or target paths.
- Accidentally implying live authority because the upstream sequence had a
  supervised fader/gain execution.

## Test plan

Run focused unit tests for not-ready, valid applied writer, dry-run rejection,
forged payload rejection, registry exposure, matrix exposure, and Product Layer
dispatch through writer apply into read-only memory index.

## Where the other agent may disagree

Kimi may argue that even a verified writer apply result should not emit an index
request until the memory file itself is read. That is a reasonable stricter
position; the compromise is that the consumer emits only a request template, and
the actual index builder remains the authority that reads and validates the
append-only memory file.
