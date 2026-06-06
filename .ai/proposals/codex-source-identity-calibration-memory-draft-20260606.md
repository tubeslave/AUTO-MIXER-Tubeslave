# Proposal: source-identity-calibration-memory-draft-20260606

## Short Thesis

Add a read-only append-only memory draft surface for accepted source identity
calibration persistence candidates, without writing memory or runtime state.

## What I Understood

The Automixer needs long-term learning from accepted source identity calibration
to improve future decisions. Directly writing `.ai/memory`, classifier models,
or downstream priority from a persistence candidate would be premature. The
right next step is a draft record that is auditable, append-only in shape, and
explicitly non-mutating.

## Proposed Solution

- Add `source_identity_calibration_memory_draft.py`.
- Revalidate the persistence plan envelope, safety flags, status, generated
  timestamp, summary counts, candidate IDs, source-type summaries, and candidate
  false-authority flags.
- Emit draft records with target memory path, append-only marker, source review
  evidence, draft text, and all write flags false.
- Publish draft status in `main_pipeline_preview`.
- Register the draft as `shadow_only` memory-learning evidence in the processing
  registry and capability matrix.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_draft.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_draft.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-memory-draft-20260606.md`

## Alternatives Considered

- Append directly to `.ai/memory/project.md`. Rejected because this needs a
  separate audited append-only write contract.
- Fold draft records into the persistence plan. Rejected because memory drafting
  should be a distinct boundary before writes.
- Wait for a writer. Rejected because the writer should consume validated draft
  records, not raw persistence candidates.

## Risks

- Draft records may be mistaken for already-written memory. Mitigate with
  explicit `memory_write_allowed_by_draft=false` and
  `memory_mutation_allowed_by_draft=false` flags.
- Forged persistence plans could create misleading draft text. Mitigate by
  revalidating envelope, summary, and candidate authority fields.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_draft.py tests/test_source_identity_calibration_persistence_plan.py tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_draft.py backend/source_identity_calibration_persistence_plan.py backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might push to implement the append-only writer now. I would keep this step
as a draft boundary because it gives the future writer a small, testable input
contract and avoids conflating review evidence with mutation.
