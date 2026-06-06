# Proposal: source-identity-proposal-queue-ordering-plan-20260606

## Short Thesis

Add a read-only source identity proposal queue-ordering plan so recommended
proposal order can become explicit import candidates without mutating the
operator queue or live authority.

## What I Understood

Ordering preview is still only a view. The next useful boundary is a plan that
names the queue order the system would like to use, validates the upstream view,
and still refuses to become a queue importer.

## Proposed Solution

- Add `source_identity_proposal_queue_ordering_plan.py`.
- Validate ordering-preview type, method, status, UTC timestamp, summary counts,
  row flags, duplicate IDs, and recommended index sequence.
- Emit `queue_ordering_plan_records` with planned queue indexes, source preview
  links, operator-review requirements, and idempotency-key requirements.
- Keep `queue_import_allowed_by_plan=false`,
  `queue_reorder_allowed_by_plan=false`, `queue_mutation=false`,
  `proposal_order_mutation=false`, `can_import_to_queue=false`, and
  `can_apply_live=false`.
- Wire Product Layer preview and register the method as `shadow_only` decision
  evidence.

## Likely Files To Touch

- `backend/source_identity_proposal_queue_ordering_plan.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_proposal_queue_ordering_plan.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-proposal-queue-ordering-plan-20260606.md`

## Alternatives Considered

- Mutate `operator_proposal_queue` directly. Rejected because queue mutation
  needs an acknowledged import consumer with override and rollback behavior.
- Reuse ordering preview as the queue plan. Rejected because preview and import
  planning are separate trust boundaries.
- Trust ordering-preview rows without revalidation. Rejected because consumer
  boundaries must fail closed on forged authority.

## Risks

- Operators may mistake plan records for queued tasks. Mitigate with explicit
  `queue_ordering_plan_only=true`, `queue_import_allowed_by_plan=false`, and
  `can_import_to_queue=false`.
- Plan records may drift from preview rows. Mitigate by validating ordered IDs,
  duplicate IDs, and the recommended index sequence.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_queue_ordering_plan.py tests/test_source_identity_proposal_ordering_preview.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_queue_ordering_plan.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue that this plan should immediately import into the operator queue.
I would keep import as the next reviewed consumer because queue mutation needs
operator acknowledgement, idempotency behavior, and rollback/override semantics.
