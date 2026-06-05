# scene-preset-cache-freshness-20260605

## Context

The long-term Automixer goal requires reading scene, snapshot, and user-preset
state before the system can safely recommend or eventually manage scene/preset
changes. The proposal layer already blocks stale scene/preset current-state
evidence, but the Product Layer snapshot/preset DTOs only expose cached catalog
coverage and do not publish cache-freshness status.

## Goal

Expose read-only cache freshness for snapshots/scenes and user presets so
scene/preset recommendations can distinguish fresh cached state from stale or
missing cached state without querying or mutating the console.

## Non-goals

- Do not load, save, recall, or store scenes/presets.
- Do not add queue/import/apply authority for scene/preset proposals.
- Do not claim physical readback proof or Full Autonomous readiness.

## Safety Constraints

- Product Layer snapshots must remain cached-only and read-only.
- `find_snap_by_name()` remains a cache/name lookup boundary, not a load path.
- Scene/preset apply remains blocked outside the narrow supervised fader/gain
  executor.

## Expected Evidence

- `snapshots_scenes_snapshot` publishes `cache_freshness` and
  `summary.cache_freshness_status`.
- `user_presets_snapshot` publishes `cache_freshness` and
  `summary.cache_freshness_status`.
- Fixture evidence records stale-cache detection for both read surfaces while
  still reporting physical readback and diff/rollback gaps.
- Focused tests prove fresh, stale, and missing cached states without mixer
  sends.
