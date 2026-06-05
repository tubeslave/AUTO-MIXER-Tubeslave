# ADR: promotion-readiness-evidence-requirements-20260605

## Context

The autonomy readiness snapshot is the Product Layer view of what can move toward
supervised or autonomous operation. It already reports blockers and next actions,
but those are prose. Scene/preset and routing now need stable requirement IDs so
agents and UI surfaces can reason about missing evidence without parsing text.

## Options considered

1. Add `evidence_requirement_ids` directly to readiness targets and rows.
2. Keep free-form text only.
3. Create a separate evidence ontology now.

## Decision

Use option 1. Promotion readiness rows will serialize stable evidence requirement
IDs next to existing blockers and next steps.

## Why this won

This is the smallest contract change that improves explainability and future
automation planning while remaining read-only and fail-closed.

## Rejected alternatives

Free-form text is too weak for autonomous planning. A separate ontology is useful
later, but premature before more rows require relationships and severity.

## Implementation plan

1. Add `evidence_requirement_ids` to `PromotionTarget`.
2. Add the field to `PromotionReadinessRow` and serialized payloads.
3. Populate routing and scene/preset target evidence IDs.
4. Assert scene/preset and routing IDs in readiness tests.
5. Keep existing live-write and autonomous blockers unchanged.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_scene_preset_management_proposal.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: consumers might interpret requirement IDs as proof status. The existing
status, blockers, and live flags remain authoritative.

Rollback is removing the field, target values, tests, and this ADR.
