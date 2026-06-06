# ADR: source-identity-proposal-queue-ordering-plan-20260606

## Context

The source identity proposal ordering preview can render a recommended proposal
order. The project still lacks a separate reviewed plan that turns that view
into explicit queue-ordering/import candidates without mutating the actual
operator queue.

## Options Considered

1. Mutate `operator_proposal_queue` directly from ordering preview.
2. Treat ordering preview rows as queue import records.
3. Add a separate read-only queue-ordering plan.

## Decision

Use option 3. Add a non-live queue-ordering plan that consumes ordering-preview
rows, validates them again, and emits separate plan records for a future
acknowledged queue import consumer.

## Why This Won

It advances autonomous decision routing while preserving the distinction between
recommended order, queue-import planning, actual queue mutation, and live
execution.

## Rejected Alternatives

- Direct queue mutation was rejected because queue changes affect operator
  attention and need acknowledgement, idempotency, override, and rollback rules.
- Reusing ordering preview rows was rejected because preview and queue import
  planning are separate trust boundaries.
- Skipping validation was rejected because forged preview rows could otherwise
  imply queue or live authority.

## Implementation Plan

1. Add the backend source identity proposal queue-ordering plan module.
2. Validate ordering-preview envelope, summary, rows, duplicate IDs, recommended
   index sequence, and safety flags.
3. Emit read-only plan records with planned queue indexes.
4. Keep Product Layer `proposals` and `operator_proposal_queue` non-mutating.
5. Register the method as shadow-only decision evidence.
6. Add focused queue-ordering plan, Product Layer, registry, and capability
   tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_queue_ordering_plan.py tests/test_source_identity_proposal_ordering_preview.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_queue_ordering_plan.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: plan records could be treated as queued tasks. Mitigation: records and
summaries keep `queue_ordering_plan_only=true`,
`queue_import_allowed_by_plan=false`, `queue_reorder_allowed_by_plan=false`,
`queue_mutation=false`, and `can_import_to_queue=false`.

Risk: ordering preview and queue plan could drift. Mitigation: plan validation
checks ordered proposal IDs, duplicate IDs, and recommended index sequence.

Rollback: remove the queue-ordering plan call from
`processing_pipeline_preview.py`, remove registry/capability rows, and keep the
ordering preview as the terminal source identity order evidence artifact.
