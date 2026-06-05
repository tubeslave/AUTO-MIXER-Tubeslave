# scene-preset-readiness-split-20260605

## Context

Autonomy promotion readiness currently groups routing, snapshots, scenes, and
presets into one target. That is too coarse for the long-term AI Mixing Engineer
goal because routing and scene/preset control have different Product Layer
evidence contracts, read surfaces, and rollback requirements.

## Goal

Split scene/preset readiness into its own fail-closed promotion target. The
target must reference scene/preset capabilities and Product Layer scene/preset
read surfaces without widening any live write authority.

## Non-goals

- Do not promote scenes or presets to supervised live execution.
- Do not add scene, snapshot, preset, or routing writes.
- Do not query, load, save, recall, or store console scenes/presets.
- Do not change safety coverage status.

## Safety constraints

- Full Autonomous remains blocked.
- Scene/preset readiness remains manual-only/proof-blocked until safety, read
  coverage, rollback, and hardware proof are complete.
- Existing routing readiness remains visible as a separate target.

## Expected evidence

- Promotion readiness has distinct routing and scene/preset target IDs.
- Scene/preset target references `console_scene_preset_control`,
  `snapshots_scenes`, and `user_presets`.
- Routing target references `console_routing_control` and `routing_and_sends`.
- Both targets remain not autonomous-ready and not supervised-live allowed.
- Tests prove no live authority widening.
