# Proposal: source-identity-calibration-memory-writer-20260606

## Short Thesis

Add an append-only memory writer contract for source identity calibration drafts,
with dry-run preview and explicit operator acknowledgement for actual file
writes.

## What I Understood

The Automixer needs durable learning from accepted source identity calibration,
but memory writes must be separated from model changes, downstream proposal
priority, queues, and live console authority. The existing draft surface defines
what should be written; the next step is a writer that validates that draft and
applies it only under an explicit operator-acknowledged path.

## Proposed Solution

- Add `source_identity_calibration_memory_writer.py`.
- Revalidate memory draft envelope, status, UTC timestamp, safety flags, summary
  counts, candidate IDs, source types, target path, and draft-record false
  authority flags.
- Return dry-run status by default.
- Require `apply=true` and `operator_ack=true` before appending to memory.
- Wire dry-run status into `main_pipeline_preview`.
- Register the writer as `shadow_only` memory-learning evidence.

## Likely Files To Touch

- `backend/source_identity_calibration_memory_writer.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_calibration_memory_writer.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-calibration-memory-writer-20260606.md`

## Alternatives Considered

- Keep draft records forever without a writer. Rejected because long-term
  learning needs an audited append path.
- Auto-apply memory drafts from preview. Rejected because preview must remain
  read-only.
- Let memory writes change downstream priority immediately. Rejected because
  runtime trust changes need a separate promotion contract.

## Risks

- File writes may be mistaken for runtime learning authority. Mitigate by naming
  the write as append-only project memory evidence and preserving runtime
  mutation flags as false.
- Unsafe draft records could be written. Mitigate by revalidating draft envelope,
  summary, target path, and record authority flags before apply.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_writer.py tests/test_source_identity_calibration_memory_draft.py tests/test_source_identity_calibration_persistence_plan.py tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_writer.py backend/source_identity_calibration_memory_draft.py backend/source_identity_calibration_persistence_plan.py backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where The Other Agent May Disagree

Kimi might argue that a writer should also update runtime trust state. I would
keep this writer file-only because runtime trust and downstream priority need a
separate promotion path with replay and rollback evidence.
