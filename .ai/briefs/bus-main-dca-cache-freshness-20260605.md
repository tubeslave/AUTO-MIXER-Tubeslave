# Bus Main DCA Cache Freshness - 2026-06-05

## Task

Add aggregate cache freshness evidence to the Product Layer bus/main/DCA snapshot without querying or mutating the console.

## Context

- Full autonomous mixing needs bus, main, and DCA state that is present and fresh enough to trust for balance, grouping, and master decisions.
- `build_bus_main_dca_snapshot` exposes cached bus, main, and DCA DTOs with per-section read coverage.
- `read_surface_inventory.bus_group_main_dca_state` still lists stale-cache timestamps as a blocker.
- The repository already uses `state_timestamps` freshness evidence for channel, matrix, routing/sends, and dLive DTOs.

## Constraints

- Do not call WING/dLive, `send`, or any bus/main/DCA setter.
- Do not promote bus, main, DCA, or grouping writes into the supervised executor.
- Keep `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`.
- Freshness must distinguish missing, unknown, stale, and fresh cache evidence.

## Expected Output

- `bus_main_dca_snapshot.summary` includes `cache_freshness_status`.
- `bus_main_dca_snapshot` includes top-level `cache_freshness` over normalized bus, main, and DCA fields.
- Read-surface inventory and fixture evidence mention bus/main/DCA cache freshness while retaining physical proof, full group membership, matrix send, and parity blockers.
- Tests prove fresh, missing, and stale behavior without mixer calls.
