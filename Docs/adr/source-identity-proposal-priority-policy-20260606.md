# ADR: source-identity-proposal-priority-policy-20260606

## Context

The source identity priority gate can mark verified rollback records as eligible
for downstream priority policy. The project still lacks a reviewed policy layer
that turns eligibility into explainable proposal-ranking evidence.

## Options Considered

1. Sort Product Layer preview proposals directly.
2. Import eligible source identity records into the operator proposal queue.
3. Add a separate read-only proposal-priority policy projection.

## Decision

Use option 3. Add a non-live policy projection that consumes eligible priority
gate records and preview proposals, then emits ranking records without changing
proposal order or queue contents.

## Why This Won

It turns trusted source identity memory into a concrete decision signal while
preserving the distinction between evidence, policy projection, queue ordering,
and live execution.

## Rejected Alternatives

- Direct sorting was rejected because proposal order is operator-facing runtime
  behavior and needs its own consumer contract.
- Queue import was rejected because ranked policy evidence is not an operator
  task and must not create pending actions.
- Skipping priority-gate validation was rejected because each policy layer must
  fail closed on forged authority.

## Implementation Plan

1. Add the backend source identity proposal-priority policy module.
2. Validate priority-gate envelope, summary, eligible records, duplicate IDs,
   and safety flags.
3. Match eligible source IDs against preview proposals.
4. Emit read-only ranked policy records and unmatched eligible source IDs.
5. Keep Product Layer preview non-mutating.
6. Register the method as shadow-only decision evidence.
7. Add focused policy, Product Layer, registry, and capability tests.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_priority_policy.py tests/test_source_identity_calibration_memory_trust_promotion_priority_gate.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_priority_policy.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Risk: policy records could be treated as actual queue ordering. Mitigation:
records and summaries keep `proposal_order_mutation=false`,
`queue_mutation=false`, `downstream_priority_change_allowed_by_policy=false`,
`live_widening_allowed=false`, and `can_import_to_queue=false`.

Risk: source matching misses useful proposals. Mitigation: the policy exposes
unmatched eligible source IDs so future proposal schemas can add explicit source
identity fields.

Rollback: remove the policy call from `processing_pipeline_preview.py`, remove
registry/capability rows, and keep the priority gate as the terminal source
identity policy evidence artifact.
