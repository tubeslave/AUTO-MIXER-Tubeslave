# scene-preset-snapshot-freshness-bridge-20260605

## Context

The Product Layer now exposes cache freshness for `snapshots_scenes_snapshot` and
`user_presets_snapshot`. The scene/preset proposal layer already blocks stale
`current_state_cache_freshness`, but callers still have to copy freshness from
Product Layer DTOs manually.

## Goal

Let scene/preset proposal analysis consume read-only Product Layer snapshot
freshness evidence when it is provided, so stale scene/preset DTOs cannot produce
proposal-only recommendations through the batch path.

## Non-goals

- Do not import Product Layer snapshot builders into the proposal module.
- Do not query the mixer or inspect hardware.
- Do not make scene/preset proposals queueable or live-applicable.

## Safety Constraints

- Explicit `current_state_cache_freshness` on an analysis input remains the
  highest-priority caller evidence.
- Shared Product Layer snapshots are optional and read-only.
- Missing snapshot evidence preserves the existing `not_provided` behavior.
- Stale or unknown provided snapshot freshness must keep recommendations
  `analysis_only`.

## Expected Evidence

- Fresh `snapshots_scenes_snapshot` evidence can support a proposal-only scene
  recommendation when all other scene/preset gates pass.
- Stale `snapshots_scenes_snapshot` evidence blocks both analyzer and batch
  proposal output.
- Stale `user_presets_snapshot` evidence blocks user-preset proposals.
- Registry/contract metadata names the Product Layer snapshot freshness bridge.
