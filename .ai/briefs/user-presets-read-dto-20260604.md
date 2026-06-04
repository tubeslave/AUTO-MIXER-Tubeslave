# Brief: user-presets-read-dto-20260604

## Context

User presets are part of the target console-control surface, but recalling or
storing a preset can change many console parameters. The read inventory still
listed this surface as missing.

## Task

Expose a Product Layer user-preset catalog DTO using cached runtime state only.
The DTO must not query WING, recall presets, store presets, or widen live write
authority.

## Safety Boundaries

- No live WING/OSC writes.
- Do not add preset recall/store handlers.
- Do not call transport helpers during DTO construction.
- Keep preset recall/store blocked until scoped diff, parameter contents, and
  rollback evidence exist.

## Expected Evidence

- Focused tests prove cached catalog exposure and empty-cache behavior.
- Handler round trip returns read-only safety flags.
- Read surface inventory moves `user_presets` from `missing` to `partial`.
