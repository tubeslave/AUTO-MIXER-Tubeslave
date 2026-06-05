# Matrix Sends Read DTO - 2026-06-05

## Task

Expose cached channel-to-matrix send state in the Product Layer routing/sends snapshot without querying or mutating the console.

## Context

- Full autonomous mixing needs visibility into how channels feed matrices before it can reason about output zones, IEM/stem routing, and matrix-level optimization.
- `build_routing_sends_snapshot` already exposes static Dante routes plus cached channel bus sends, channel main sends, and output route source fields.
- `read_surface_inventory.matrix_state` still lists matrix sends as a matrix-state blocker.
- WING docs in this repo identify matrix send cache addresses under `/ch/X/send/MX1/on` and `/ch/X/send/MX1/lvl`.

## Constraints

- Do not call WING/dLive, `send`, or any routing setter.
- Do not add matrix-send write proposals or queue actions.
- Keep `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`.
- Missing matrix send cache must be visible through coverage flags instead of silently omitted.

## Expected Output

- `routing_sends_snapshot` includes `channel_matrix_sends` rows.
- Summary and read coverage count matrix send rows separately.
- Read-surface inventory and fixture evidence mention cached matrix sends while retaining physical proof blockers.
- Tests prove present and missing cached matrix-send behavior without mixer calls.
