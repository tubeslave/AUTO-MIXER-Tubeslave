# Brief: fx-inserts-read-dto-20260605

## Context

FX rack slots and inserts are required for the long-term autonomous console
control target. Existing helpers can query FX and insert state, but the Product
Layer did not expose a stable cached read DTO for cockpit display.

## Task

Expose a Product Layer FX/insert snapshot using cached runtime state only. The
DTO must not query WING, change FX models, toggle FX, change mix, assign inserts,
or widen live write authority.

## Safety Boundaries

- No live WING/OSC writes.
- Do not call `query_fx_slot`, `query_insert`, `get_fx_overview`, or
  `get_insert_state` while building the DTO.
- Keep FX and insert apply blocked until parameter schemas, readback fixtures,
  scoped diff, and rollback proof exist.

## Expected Evidence

- Focused tests prove cached FX/insert exposure and empty-cache behavior.
- Handler round trip returns read-only safety flags.
- Read surface inventory marks FX/insert coverage as `partial`, not live-ready.
