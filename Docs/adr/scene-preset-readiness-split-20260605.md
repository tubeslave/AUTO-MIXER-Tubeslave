# ADR: scene-preset-readiness-split-20260605

## Context

The autonomy promotion readiness matrix had one combined
`routing_scene_preset_supervised_promotion` target. Routing and scene/preset
control both remain unsafe for live automation, but they now have different
read-only evidence surfaces. Scene/preset work has its own snapshot freshness,
scoped diff, and read-only rollback evidence contracts.

## Options considered

1. Split routing and scene/preset readiness targets while keeping both blocked.
2. Keep the combined target.
3. Create a new scene/preset safety coverage row immediately.

## Decision

Use option 1. The readiness matrix will expose separate routing and scene/preset
targets. Both targets remain fail-closed and share the existing blocked
`routing_snapshots_scenes` safety row for now.

## Why this won

The split improves evidence visibility without implying live readiness. It lets
future work advance routing and scene/preset proof independently while keeping
the current live executor restricted to supervised fader/gain.

## Rejected alternatives

Keeping the combined target hides scene/preset-specific progress. Creating a new
safety row now would suggest a stronger proof boundary than the repo currently
has.

## Implementation plan

1. Replace the combined promotion target with `routing_supervised_promotion`.
2. Add `scene_preset_supervised_promotion`.
3. Point routing to `routing_and_sends`.
4. Point scene/preset to `snapshots_scenes` and `user_presets`.
5. Update tests to assert both rows remain blocked and non-live.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_autonomy_promotion_readiness.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_scene_preset_management_proposal.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: target split could be mistaken for promotion. The implementation keeps
`supervised_live_allowed=False`, `autonomous_ready=False`, the blocked safety row,
and the quarantined write surface.

Rollback is restoring the single combined target and its tests.
