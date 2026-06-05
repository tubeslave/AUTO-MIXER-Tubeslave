# Brief: scene-preset-management-freshness-gate-20260605

## Context

The project goal requires scene and preset control, but current live authority
does not include scene recall, snapshot recall, preset recall, preset save, or
broad console-state writes. The scene/preset proposal layer uses current state
readback, target previews, scoped diffs, rollback snapshots, scope limits,
activity guards, and operator intent. It also needs to respect whether the
current scene/preset state cache is fresh enough to trust.

## Task

Add a read-only freshness gate to `scene_preset_management_proposal`: when
current scene/preset state freshness is supplied and is not fresh, demote the
recommendation to `analysis_only` with an explicit blocker. Fresh state should
remain eligible for proposal-only output.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes.
- Do not add scene, snapshot, preset recall, preset save, or broad state apply
  support.
- Keep scene/preset output proposal-only and non-queueable.
- Do not query a console from the proposal module.

## Expected Evidence

- Fresh current-state freshness is recorded in scene/preset proposal evidence.
- Stale current-state freshness blocks proposal-only output.
- `build_scene_preset_management_proposals` filters stale-state items.
- Proposal-only registry/contract metadata mentions current scene/preset
  freshness.
