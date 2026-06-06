# Proposal: source-identity-calibration-memory-index-20260606

## Short Thesis

Add a read-only source identity calibration memory index that turns appended
memory blocks into validated review evidence without granting runtime trust or
downstream priority authority.

## What I Understood

The Automixer needs durable learning, but the boundary between "memory exists"
and "memory may influence autonomous decisions" must stay explicit. The writer
can append operator-acknowledged memory records; an index should make those
records visible for review while preserving the need for separate replay/trust
promotion before downstream proposal behavior changes.

## Proposed Solution

- Add `source_identity_calibration_memory_index.py`.
- Parse only the known append-only writer block format.
- Validate required fields, UTC review timestamp, source identity draft ID
  prefix, exact runtime-authority line, and approved memory target path.
- Return fail-closed replacement status for malformed blocks.
- Wire index status and counts into `main_pipeline_preview`.
- Register the index as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_index.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_index.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-memory-index-20260606.md`

## Alternatives Considered

- Let downstream gates read `.ai/memory/project.md` directly. Rejected because
  downstream trust needs a validated evidence surface, not ad hoc file parsing.
- Promote indexed memory directly into source identity trust. Rejected because
  memory existence is not enough to change proposal priority.
- Store records in a new database. Rejected because this slice should keep the
  existing append-only project memory contract and avoid dependencies.

## Risks

- Parsed memory could be mistaken for authority. Mitigation: index records force
  `usable_for_downstream_priority=false` and preserve runtime/model/queue/live
  mutation flags as false.
- Markdown parsing could drift from writer output. Mitigation: tests assert the
  exact block fields and runtime-authority line.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_index.py tests/test_source_identity_calibration_memory_writer.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_index.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue that the index should immediately feed
`source_identity_downstream_gate`. I would keep that as a later promotion step
because indexed memory is evidence, not proof that the current source context is
fresh enough for downstream priority changes.
