# Proposal: source-identity-proposal-priority-policy-20260606

## Short Thesis

Add a read-only source identity proposal-priority policy projection so trusted
source identity memory can produce explainable ranking evidence without
reordering proposals, mutating queues, or widening live authority.

## What I Understood

The priority gate can say that a verified source identity record is eligible for
downstream priority policy. The next project step is not live control; it is a
transparent policy consumer boundary that shows which preview proposals would be
ranked higher because their source identity is trusted.

## Proposed Solution

- Add `source_identity_proposal_priority_policy.py`.
- Validate priority-gate type, method, status, UTC timestamp, summary counts,
  safety flags, and eligible priority records.
- Match eligible `source_candidate_id` records against preview proposals using
  proposal target/source fields.
- Emit read-only policy records with base score, source-identity adjustment,
  adjusted score, and recommended rank bucket.
- Keep `proposal_order_mutation=false`, `queue_mutation=false`,
  `direct_downstream_priority_mutation=false`,
  `downstream_priority_change_allowed_by_policy=false`,
  `live_widening_allowed=false`, `can_import_to_queue=false`, and
  `can_apply_live=false`.
- Wire Product Layer preview and register the method as `shadow_only` decision
  evidence.

## Likely Files To Touch

- `backend/source_identity_proposal_priority_policy.py`
- `backend/processing_pipeline_preview.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_source_identity_proposal_priority_policy.py`
- `tests/test_operator_product_state.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-proposal-priority-policy-20260606.md`

## Alternatives Considered

- Sort Product Layer proposals immediately. Rejected because ordering mutation
  affects operator behavior and needs a separate consumer contract.
- Import ranked records into the decision queue. Rejected because this policy is
  analysis evidence, not an operator task.
- Trust priority-gate records without revalidation. Rejected because policy
  layers must reject forged authority independently.

## Risks

- Ranking evidence may be mistaken for queue ordering authority. Mitigate with
  explicit `policy_evidence_only=true`, `proposal_order_mutation=false`, and
  `live_widening_allowed=false`.
- Matching source IDs across heterogeneous proposals may miss some useful
  candidates. Mitigate by keeping unmatched eligible source IDs visible.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_proposal_priority_policy.py tests/test_source_identity_calibration_memory_trust_promotion_priority_gate.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m py_compile backend/source_identity_proposal_priority_policy.py backend/processing_pipeline_preview.py backend/processing_registry.py backend/project_goal_capability_matrix.py`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue that the policy should directly sort preview proposals. I would
keep sorting as the next separate consumer because proposal order changes affect
operator attention and need an auditable rollback and override story.
