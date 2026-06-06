# ADR: source-identity-calibration-memory-draft-20260606

## Context

Source identity calibration now has preview candidates, an acceptance ledger, and
a manual persistence plan. The next gap toward long-term learning is to express
accepted persistence candidates as append-only memory drafts without actually
writing memory, models, downstream priority, queues, or live console state.

## Options Considered

1. Append accepted calibration records directly to `.ai/memory/project.md`.
2. Keep persistence candidates only in the persistence plan.
3. Add a separate read-only source identity calibration memory draft.

## Decision

Use option 3. Add a transport-free memory draft builder that validates the
persistence plan and surfaces append-only draft records with no write authority.

## Why This Won

The draft makes the future memory update contract concrete while preserving a
clear boundary: persistence candidates are evidence, draft records are proposed
append-only memory text, and an eventual writer must still be separately audited
before mutating files or runtime trust state.

## Rejected Alternatives

- Direct memory append was rejected because this slice does not define the full
  append-only writer, rollback, or review contract.
- Plan-only persistence was rejected because future memory work needs a stable
  draft payload, not raw plan candidates.
- A live/runtime trust update was rejected because source identity memory must
  not change proposal priority until a separate audited write contract exists.

## Implementation Plan

1. Add a backend draft module with strict persistence-plan validation.
2. Validate plan envelope, status, UTC timestamps, safety flags, summary counts,
   summary identity lists, summary safety claims, and candidate false-authority
   flags.
3. Emit append-only draft records only when the plan is clean and candidates are
   present.
4. Wire the draft into `main_pipeline_preview`.
5. Register the draft as `shadow_only` memory-learning evidence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_draft.py tests/test_source_identity_calibration_persistence_plan.py tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_draft.py backend/source_identity_calibration_persistence_plan.py backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: draft records could be mistaken for persisted memory. Mitigation: draft
safety and summary flags force `append_only_memory_write_allowed_by_draft=false`,
`model_mutation_allowed_by_draft=false`, `memory_mutation_allowed_by_draft=false`,
`downstream_priority_change_allowed_by_draft=false`, and `live_write=false`.

Rollback: remove the draft call from `processing_pipeline_preview.py`, remove
the registry/capability rows, and keep persistence candidates as read-only plan
evidence only.
