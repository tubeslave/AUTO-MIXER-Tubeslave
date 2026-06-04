# Brief: snapshots-scenes-read-dto-20260604

## Context

Snapshots/scenes are broad console-state surfaces. Loading a snapshot can
overwrite live mixer state, and saving can persist or replace a console scene.
The current safe baseline is read/proposal-only for this family.

## Task

Expose a Product Layer read DTO for snapshots/scenes using cached runtime state
only. The DTO must not query WING, load a scene, save a scene, or widen live
write authority.

## Safety Boundaries

- No live WING/OSC writes.
- Do not call snapshot load/save handlers.
- Do not call `find_snap_by_name()` or any helper that sends OSC during DTO
  construction.
- Keep scene load/save automation blocked until scene diff and rollback proof
  exist.

## Expected Evidence

- Focused unit tests prove cached catalog exposure and empty-cache behavior.
- Handler round trip returns read-only safety flags.
- Read surface inventory remains `partial`, with blocker text for scene diff and
  rollback evidence.
