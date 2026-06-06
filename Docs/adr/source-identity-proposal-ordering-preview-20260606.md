# ADR: source-identity-proposal-ordering-preview-20260606

## Context

The source identity proposal-priority policy can emit ranked policy records from
trusted source identity evidence. The project still lacks a separate reviewed
view that shows a recommended proposal order without changing the Product Layer
proposal payload or operator queue.

## Options Considered

1. Sort Product Layer preview proposals directly.
2. Import ranked policy records into the operator proposal queue.
3. Add a separate read-only proposal ordering preview.

## Decision

Use option 3. Add a non-live ordering preview that consumes the priority-policy
payload and preview proposals, validates the evidence again, and emits a
separate recommended ordered view.

## Why This Won

It makes trusted source identity ranking evidence visible to the operator while
preserving the distinction between evidence, recommended order, queue ordering,
and live execution.

## Rejected Alternatives

- Direct sorting was rejected because proposal order is operator-facing runtime
  behavior and needs its own consumer contract.
- Queue import was rejected because recommended-order rows are not operator
  tasks and must not create pending actions.
- Skipping priority-policy validation was rejected because stale or forged
  policy records could otherwise imply authority the policy did not grant.

## Implementation Plan

1. Add the backend source identity proposal ordering-preview module.
2. Validate priority-policy envelope, summary, policy records, duplicate IDs,
   proposal links, and safety flags.
3. Emit ranked records first and append unranked proposals in original order.
4. Keep Product Layer `proposals` non-mutating.
5. Register the method as shadow-only decision evidence.
6. Add focused ordering-preview, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_ordering_preview.py tests/test_source_identity_proposal_priority_policy.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_ordering_preview.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: ordering-preview records could be treated as actual queue ordering.
Mitigation: rows and summaries keep `ordering_preview_only=true`,
`proposal_order_mutation=false`, `queue_mutation=false`,
`downstream_priority_change_allowed_by_policy=false`,
`live_widening_allowed=false`, and `can_import_to_queue=false`.

Risk: stale policy records could reference an old proposal payload. Mitigation:
records must link back to current proposal IDs and original indexes.

Rollback: remove the ordering-preview call from `processing_pipeline_preview.py`,
remove registry/capability rows, and keep the proposal-priority policy as the
terminal source identity ordering evidence artifact.
