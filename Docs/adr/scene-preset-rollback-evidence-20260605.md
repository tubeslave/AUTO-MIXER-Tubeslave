# ADR: scene-preset-rollback-evidence-20260605

## Context

Scene/preset proposals already require rollback availability and ID, but those
fields do not distinguish read-only rollback evidence from an executable rollback
operation. Broad scene/preset control needs rollback proof as structured evidence
before any future promotion can be considered.

## Options considered

1. Add a read-only rollback evidence payload.
2. Keep rollback as boolean plus ID.
3. Run rollback validation from the proposal module.

## Decision

Use option 1. `ScenePresetManagementAnalysisInput` accepts optional
`rollback_evidence`, and the analyzer derives rollback availability, ID,
confidence, and target coverage from it.

## Why this won

This keeps proposal analysis transport-free while making rollback proof explicit
and fail-closed. Unsafe rollback payloads cannot produce proposal-only
recommendations.

## Rejected alternatives

Boolean-only rollback was too weak for future autonomous promotion. Runtime
rollback validation was rejected because the proposal module must not query or
mutate hardware.

## Implementation plan

1. Add optional rollback evidence to the analysis input.
2. Normalize rollback payload fields and safety flags.
3. Use rollback evidence as a source for rollback availability, ID, and
   confidence.
4. Add blockers for live/query/mutating rollback payloads and target mismatch.
5. Update registry/contract metadata and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/scene_preset_management_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_scene_preset_management_proposal.py tests/test_project_goal_capability_matrix.py`
- Focused scene/preset proposal tests.
- Registry/capability tests.
- Standard project pytest command.

## Risks and rollback

Risk: rollback evidence could be interpreted as permission to execute rollback.
The payload remains evidence only; proposals still have `can_apply_live=False`
and `can_import_to_queue=False`.

Rollback is removing the evidence field, normalizer, blockers, metadata, tests,
and this ADR. No live write path is introduced.
