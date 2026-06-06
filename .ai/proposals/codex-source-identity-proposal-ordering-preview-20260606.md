# Proposal: source-identity-proposal-ordering-preview-20260606

## Short Thesis

Add a read-only source identity proposal ordering preview so trusted source
identity ranking evidence can be inspected as a recommended order without
mutating proposal lists, queues, or live authority.

## What I Understood

The priority policy already projects ranked policy records. The next useful
step is a consumer boundary that renders an ordered view, while still refusing
to become the actual queue order, queue importer, downstream priority mutator,
or live executor.

## Proposed Solution

- Add `source_identity_proposal_ordering_preview.py`.
- Validate priority-policy type, method, status, UTC timestamp, summary counts,
  policy records, duplicate IDs, proposal links, and safety flags.
- Emit `ordered_proposals` as a separate read-only view:
  ranked policy records first, unranked proposals afterward in original order.
- Keep the original Product Layer `proposals` array unchanged.
- Keep `proposal_order_mutation=false`, `queue_mutation=false`,
  `direct_downstream_priority_mutation=false`,
  `downstream_priority_change_allowed_by_policy=false`,
  `live_widening_allowed=false`, `can_import_to_queue=false`, and
  `can_apply_live=false`.
- Wire Product Layer preview and register the method as `shadow_only` decision
  evidence.

## Likely Files To Touch

- `backend/source_identity_proposal_ordering_preview.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_proposal_ordering_preview.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-proposal-ordering-preview-20260606.md`

## Alternatives Considered

- Sort Product Layer proposals directly. Rejected because this would change
  operator-facing runtime behavior and needs its own mutation contract.
- Import ordered rows into the decision queue. Rejected because ordering preview
  is evidence, not an operator task.
- Trust priority-policy records without revalidation. Rejected because consumer
  boundaries must fail closed on forged authority.

## Risks

- The preview may be mistaken for actual queue ordering. Mitigate with explicit
  `ordering_preview_only=true`, `proposal_order_mutation=false`, and
  `can_import_to_queue=false`.
- Stale priority-policy records may reference old proposals. Mitigate by
  validating proposal IDs and original indexes against the current proposal
  payload.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_ordering_preview.py tests/test_source_identity_proposal_priority_policy.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_ordering_preview.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue that the ordering preview should become the actual operator
queue order immediately. I would keep that as a later consumer because queue
mutation needs override behavior, rollback evidence, and explicit operator
review.
