# ADR: source-identity-calibration-memory-writer-20260606

## Context

Source identity calibration now has preview candidates, an acceptance ledger, a
manual persistence plan, and append-only memory draft records. The remaining
gap is an audited writer boundary that can apply those draft records without
granting model, queue, downstream-priority, runtime-memory, or live-console
authority.

## Options Considered

1. Append memory draft records automatically from Product Layer preview.
2. Keep draft records forever without an apply path.
3. Add a separate append-only memory writer that is dry-run in preview and
   requires explicit operator acknowledgement before writing.

## Decision

Use option 3. Add a transport-free memory writer contract that validates memory
draft envelopes and records, reports dry-run readiness in preview, and appends
only to `.ai/memory/project.md` when invoked with `apply=true` and
`operator_ack=true`.

## Why This Won

The writer makes durable learning possible while preserving the live-sound
safety boundary. Preview remains read-only, file writes require an explicit
operator-acknowledged call, and the writer does not mutate source identity
models, downstream proposal priority, queues, runtime trust, mixer transport, or
console state.

## Rejected Alternatives

- Automatic preview writes were rejected because Product Layer preview must stay
  read-only and non-promoting.
- Draft-only persistence was rejected because accepted operator calibration
  needs a future durable memory path.
- Runtime trust or model updates were rejected because file memory append is not
  the same as promotion authority.

## Implementation Plan

1. Add a backend writer module with strict memory-draft validation.
2. Validate draft envelope, method, status, UTC timestamps, safety flags, summary
   counts, summary identity lists, target path, and draft-record false-authority
   flags.
3. Return dry-run readiness by default and require both `apply=true` and
   `operator_ack=true` before appending records.
4. Wire the writer into `main_pipeline_preview` as dry-run only.
5. Register the writer as `shadow_only` memory-learning evidence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_writer.py tests/test_source_identity_calibration_memory_draft.py tests/test_source_identity_calibration_persistence_plan.py tests/test_source_identity_calibration_acceptance_ledger.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_calibration_memory_writer.py backend/source_identity_calibration_memory_draft.py backend/source_identity_calibration_persistence_plan.py backend/source_identity_calibration_acceptance_ledger.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: append-only file writes could be mistaken for runtime learning authority.
Mitigation: writer safety and summary flags force `queue_mutation=false`,
`model_mutation=false`, `runtime_memory_mutation=false`,
`downstream_priority_mutation=false`, `live_authority_granted=false`, and
`full_autonomous_ready=false`.

Rollback: remove the writer call from `processing_pipeline_preview.py`, remove
the registry/capability rows, and keep memory drafts as read-only proposed text
until a replacement writer contract is reviewed.
