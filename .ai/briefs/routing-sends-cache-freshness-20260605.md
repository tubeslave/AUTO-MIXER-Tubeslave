# Routing Sends Cache Freshness - 2026-06-05

## Task

Add cache freshness evidence to the Product Layer routing/sends snapshot without querying or mutating the console.

## Context

- Full autonomous mixing needs routing/send state that is not only present, but fresh enough to trust for real-time decisions.
- `build_routing_sends_snapshot` exposes static Dante routes plus cached channel bus sends, channel main sends, channel matrix sends, and output route sources.
- `read_surface_inventory.routing_and_sends` still lists stale-cache timestamps as a blocker.
- The repository already has `state_timestamps`-based freshness helpers for channel inventory, channel processing, dLive, and matrix dynamics.

## Constraints

- Do not call WING/dLive, `send`, or any routing setter.
- Do not promote routing/send writes into the supervised executor.
- Keep `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`.
- Freshness must distinguish missing, unknown, stale, and fresh cache evidence.

## Expected Output

- Routing/sends snapshot summary includes `cache_freshness_status`.
- Bus, main, matrix send rows and output route rows expose per-row `read_coverage.cache_freshness`.
- The top-level routing/sends snapshot exposes aggregate `cache_freshness`.
- Read-surface inventory and fixture evidence mention routing/sends cache freshness while retaining physical proof and route diff blockers.
- Tests prove fresh, missing, and stale cache behavior without mixer calls.
