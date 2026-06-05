# Proposal: scene-preset-snapshot-freshness-bridge-20260605

## 1. Short thesis

Bridge cached Product Layer scene/preset snapshot freshness into
`scene_preset_management_proposal`, keeping scene and preset actions
proposal-only and non-queueable.

## 2. What I understood

Scene/preset control is required for the final autonomous AI Mixing Engineer, but
it is dangerous because recalls and stores can overwrite broad console state.
The read DTO now reports cache freshness; the proposal layer should use that
evidence instead of relying on every caller to manually copy it.

## 3. Proposed solution

Add optional snapshot evidence fields to `ScenePresetManagementAnalysisInput` and
a transport-free helper that extracts `cache_freshness` from either
`snapshots_scenes_snapshot` or `user_presets_snapshot`. The analyzer uses explicit
`current_state_cache_freshness` first, then falls back to snapshot-derived
freshness. `build_scene_preset_management_proposals()` can accept shared Product
Layer snapshots and inject them into inputs that do not already carry explicit
freshness or snapshot evidence.

## 4. Likely files to touch

- `backend/scene_preset_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_scene_preset_management_proposal.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/scene-preset-snapshot-freshness-bridge-20260605.md`
- `Docs/adr/scene-preset-snapshot-freshness-bridge-20260605.md`

## 5. Alternatives considered

- Import and build Product Layer snapshots inside the proposal module. Rejected
  because proposal modules should stay transport-free and not depend on runtime
  server state.
- Require every caller to pass freshness manually. Rejected because the new DTO
  freshness evidence would remain easy to forget.
- Treat stale snapshot freshness as acceptable if preview/diff flags are true.
  Rejected because stale scene/preset cache can invalidate the active scene,
  target identity, preview, or rollback assumptions.

## 6. Risks

The bridge may make fewer recommendations proposal-only when stale snapshots are
provided. That is acceptable; stale broad-state evidence should fail closed.

## 7. Test plan

- Compile touched Python modules.
- Run focused scene/preset proposal tests.
- Run registry/capability matrix tests for metadata.
- Run standard pytest before merge.

## 8. Where the other agent may disagree

Kimi may prefer waiting for scoped scene/preset diff artifacts before adding a
bridge. I would land the bridge now because it only consumes existing read-only
freshness evidence and prevents stale DTOs from being overtrusted.
