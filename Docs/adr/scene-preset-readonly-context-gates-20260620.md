# ADR: Scene/Preset Read-Only Context Gates

## Context

The project goal requires control of scenes and user presets. Scene and preset
actions are especially risky because they can overwrite broad console state,
change many channel parameters at once, replace FX settings, or make rollback
ambiguous if the current scene/preset inventory is stale.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep scene/preset proposals usable when current scene/preset context is
   missing.
2. Require fresh read-only/no-authority scene/preset context, scoped diffs and
   rollback evidence before proposal emission.
3. Disable scene/preset proposals entirely until WING hardware proof exists.

## Decision

Use option 2. Scene/preset proposal candidates require fresh read-only/no-
authority current scene/preset context plus target preview, scoped diff,
rollback evidence and destructive-scope limits before they can become
operator-facing proposals.

## Why This Won

It advances scene/snapshot/preset management, a separate console-control part
of the AI Mixing Engineer goal, while preserving the current live boundary.
It also handles the real failure modes for scene work: stale inventories can
target the wrong slot, broad diffs can overwrite unrelated state, and missing
rollback evidence makes recovery unverifiable.

## Rejected Alternatives

- Allowing missing context was rejected because scene and preset actions need
  current readback, target preview, scoped diff, rollback evidence, operator
  intent and destructive-scope limits.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve the future assisted workflow without touching hardware.

## Implementation Plan

1. Record the existing scene/preset proposal freshness gates as M091.
2. Require fresh current scene/preset context for proposal emission.
3. Require no-authority scene/preset context.
4. Keep missing, stale, live-authority, broad-scope or rollback-incomplete
   candidates analysis-only.
5. Record M091 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_scene_preset_management_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_goal_capability_matrix.py::test_scene_preset_control_is_tracked_as_proposal_layer_without_live_widening -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m091_scene_preset_readonly_context_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M091 commit, which
restores the previous scene/preset rail documentation and M091 state/report
rows.
