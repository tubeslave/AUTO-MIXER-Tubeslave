# Proposal: Promotion Memory Writer Apply Result Consumer

## Short thesis

Bridge promotion memory-writer dry-run evidence to memory-index request
templates by wrapping the existing generic writer-apply result consumer.

## What I understood

The generic queue-import memory path can already verify an explicitly applied
append-only memory write and emit a read-only memory-index request. The
promotion queue-import chain needs the same step while preserving promotion
provenance and keeping consumer-side writes false.

## Proposed solution

Add a promotion memory-writer apply result consumer that validates the prior
promotion dry-run result consumer, delegates writer-result validation to the
generic consumer, and emits a promotion-scoped memory-index request template
only when both layers are ready.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer.py`
- `backend/handlers/product_state_handlers.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/`

## Alternatives considered

Reuse the generic consumer directly from Product State. Rejected because the
promotion path would lose explicit provenance and could not prove that the
writer result belongs to the promotion dry-run chain.

## Risks

The writer result can represent a real append-only memory file mutation, so the
consumer must distinguish source writer mutation from consumer-side mutation.
The consumer must keep index invocation false.

## Test plan

Run py_compile, focused pytest for the new and adjacent consumers,
ProductState coverage, memory lint, full pytest, and `git diff --check`.

## Where the other agent may disagree

Kimi may argue that the chain should jump directly to the existing memory-index
consumer. The safer boundary is to add the provenance wrapper first and keep
index invocation as a separate explicit step.
