# read-surface-next-readback-target-20260605

## Context

Settings now shows compact read-surface physical gap IDs. Product Layer still
does not expose an explicit "next physical readback proof target", so frontend
consumers infer it from the first gap ID.

## Goal

Expose a normalized `next_physical_readback_surface_id` in Product Layer proof
quality and render it in Settings as the next hardware readback proof target.

## Non-goals

- Do not query WING/dLive hardware in this slice.
- Do not mark the target as proven or ready.
- Do not add live read/write controls or Full Autonomous promotion logic.

## Safety constraints

- The target is read-only planning evidence.
- `full_autonomous_ready` remains false while physical gaps exist.
- Missing target metadata must fail closed to "No target" rather than implying
  readiness.

## Expected evidence

- Product Layer payload contains `next_physical_readback_surface_id`.
- Frontend adapter carries it in module profiles and next-step text.
- Settings inspector shows `Next read proof`.
- Backend, frontend contract, and e2e tests cover the path.
