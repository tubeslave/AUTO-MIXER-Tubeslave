# Matrix Cache Freshness - 2026-06-05

## Task

Add aggregate cache freshness evidence to the Product Layer matrix snapshot without querying or mutating the console.

## Context

- Full autonomous mixing needs matrix state that is present and fresh enough to trust for output, matrix, and master decisions.
- `build_matrix_snapshot` exposes cached matrix fader/mute/pan, EQ, dynamics, and insert state.
- `read_surface_inventory.matrix_state` still lists stale-cache timestamps as a blocker.
- The repository already uses `state_timestamps` freshness evidence for channel inventory, channel processing, dLive, routing/sends, and matrix dynamics.

## Constraints

- Do not call WING/dLive, `send`, or any matrix setter.
- Do not promote matrix EQ/dynamics/insert/routing writes into the supervised executor.
- Keep `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`.
- Freshness must distinguish missing, unknown, stale, and fresh cache evidence.

## Expected Output

- `matrix_snapshot.summary` includes `cache_freshness_status`.
- `matrix_snapshot` includes top-level `cache_freshness` over matrix basic controls, EQ, dynamics, and inserts.
- Read-surface inventory and fixture evidence mention matrix cache freshness while retaining physical proof, parity, and route diff blockers.
- Tests prove fresh, missing, and stale matrix-cache behavior without mixer calls.
