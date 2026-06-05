# ADR: scene-preset-snapshot-freshness-bridge-20260605

## Context

`snapshots_scenes_snapshot` and `user_presets_snapshot` now expose read-only
cache freshness. `scene_preset_management_proposal` already blocks stale
`current_state_cache_freshness`, but without a bridge every caller must manually
copy freshness from the Product Layer DTO before scene/preset analysis.

## Options considered

1. Add an optional Product Layer snapshot freshness bridge to the proposal layer.
2. Import Product Layer builders into the proposal module.
3. Keep callers responsible for copying freshness manually.

## Decision

Use option 1. Scene/preset analysis accepts optional read-only
`snapshots_scenes_snapshot` and `user_presets_snapshot` evidence and derives
current-state freshness from the relevant snapshot when explicit freshness is not
provided.

## Why this won

This keeps the module transport-free while making stale DTO evidence fail closed.
It also makes batch proposal generation safer: one shared stale snapshot blocks
all affected scene/preset recommendations instead of producing proposal-only
payloads from outdated cached state.

## Rejected alternatives

Importing Product Layer builders was rejected because the proposal module should
not depend on server/runtime state or accidentally query hardware. Manual-only
freshness was rejected because it creates a predictable caller omission risk.

## Implementation plan

1. Add optional snapshot evidence fields to scene/preset analysis input.
2. Add a helper that extracts normalized freshness from Product Layer snapshots.
3. Teach analyzer to prefer explicit freshness and otherwise use snapshot
   freshness.
4. Teach batch builder to accept shared snapshots.
5. Update registry/contract/capability metadata and tests.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/scene_preset_management_proposal.py backend/processing_registry.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_scene_preset_management_proposal.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py`
- Focused scene/preset proposal tests.
- Registry/capability matrix tests.
- Standard project pytest command.

## Risks and rollback

Risk: stale Product Layer snapshots suppress scene/preset recommendations that
would otherwise pass. This is intentional fail-closed behavior for broad
console-state changes.

Rollback is removing the snapshot evidence fields, helper, batch-builder
parameters, tests, metadata updates, and this ADR. No live write path is
introduced.
