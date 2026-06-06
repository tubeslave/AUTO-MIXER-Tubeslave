# ADR: source-identity-queue-import-visibility-20260606

## Context

Source identity queue import can now mutate only the explicit native operator
proposal queue after `apply=true` and `operator_ack=true`. Once proposals are
pending, the Product Layer needs a compact way to show which entries came from
source identity ranking and which remain analysis-only or applyable under the
existing supervised proposal gate.

## Options Considered

1. Add visibility inside `OperatorProposalQueue.summary()`.
2. Add a separate source identity queue status handler.
3. Add read-only source identity import summary fields to `build_decision_queue`.

## Decision

Use option 3. `build_decision_queue` will compute source identity import
visibility from normalized pending/history proposal dictionaries.

## Why This Won

The decision queue is already the Product Layer contract for pending operator
actions. Computing the summary there avoids changing queue storage behavior and
keeps the safety message near the proposals the operator is reviewing.

## Rejected Alternatives

- Queue-level summary was rejected because source identity import visibility is
  a Product Layer view concern.
- A separate handler was rejected because it would split queue provenance away
  from the queue payload operators already inspect.
- Live-apply promotion was rejected because source identity import is not proof
  of physical readback, rollback, feedback safety, or broader executor
  readiness.

## Implementation Plan

1. Add a read-only helper for source identity queue-import summary.
2. Attach the helper output to `decision_queue.summary`.
3. Add top-level aliases for stable cockpit counters.
4. Test acknowledged import plus decision-queue summary visibility.
5. Record the durable boundary in `.ai/memory/project.md`.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_decision_queue_summarizes_source_identity_queue_import_visibility tests/test_operator_product_state.py::test_source_identity_queue_import_handler_imports_with_ack_without_live_write -q`
- `PYTHONPATH=backend python3 -m py_compile backend/operator_product_state.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks And Rollback

Risk: source identity queue provenance could be confused with permission to
write the console. Mitigation: the summary states that import has no live-write
authority and that later console writes remain governed by
`apply_operator_proposal`.

Rollback: remove the added decision-queue summary fields and tests. Imported
proposals will still exist in pending actions; only the compact visibility layer
will disappear.
