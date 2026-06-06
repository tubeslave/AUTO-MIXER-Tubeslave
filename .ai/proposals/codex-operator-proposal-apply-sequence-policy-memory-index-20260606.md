# Proposal: operator-proposal-apply-sequence-policy-memory-index-20260606

## Short thesis

Index appended sequence policy memory as read-only evidence before any future
policy or trust promotion step.

## What I understood

The project goal needs long-term learning from successful supervised decisions,
but the current live authority must remain limited to the existing supervised
fader/gain apply path. The new memory writer creates append-only text; the next
safe step is to parse that text into structured evidence, not to let it steer
execution.

## Proposed solution

Add `backend/operator_proposal_apply_sequence_policy_memory_index.py` with a
transport-free builder that reads either supplied markdown text or the allowed
target path `.ai/memory/project.md`, parses `### Supervised sequence policy
memory` blocks, validates their fields and safety line, and returns structured
records with all mutation/execution flags false.

## Likely files to touch

- `backend/operator_proposal_apply_sequence_policy_memory_index.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/processing_pipeline_preview.py`
- `tests/test_operator_proposal_apply_sequence_policy_memory_index.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/operator-proposal-apply-sequence-policy-memory-index-20260606.md`

## Alternatives considered

- Let the writer immediately feed policy decisions. Rejected: append-only memory
  is not policy authority.
- Keep the index outside the pipeline preview. Rejected: hidden memory evidence
  is harder to audit and less useful for operator review.

## Risks

- Memory text could be malformed or forged. The parser must fail closed and
  expose replacement-required status.
- The index could be mistaken for execution authority. Every record and summary
  keeps execution, queue, policy, model, runtime-memory, and live authority false.

## Test plan

- Parser tests for no source, valid source, temp-root read, malformed authority,
  fader ceiling, and target path rejection.
- Product Layer preview test for adapter/summary visibility.
- Registry and capability matrix checks for `shadow_only` and no executor role.

## Where the other agent may disagree

Kimi may argue the index should wait until a full trust-promotion design exists.
The counterargument is that indexing is a necessary read-only boundary and gives
future promotion work a safer, validated input surface.
