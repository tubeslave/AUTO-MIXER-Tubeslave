# FX Inserts Cache Freshness - 2026-06-05

## Task

Add aggregate cache freshness evidence to the Product Layer FX/inserts snapshot without querying or mutating the console.

## Context

- Full autonomous mixing needs FX rack and insert state that is present and fresh enough to trust before proposing FX, send, or insert changes.
- `build_fx_inserts_snapshot` exposes cached FX slot metadata, cached numeric FX parameters, and insert assignments with read coverage.
- `read_surface_inventory.fx_slots_and_inserts` still lists stale-cache timestamps as a blocker.
- The repository already uses `state_timestamps` freshness evidence for channel, matrix, routing/sends, bus/main/DCA, and dLive DTOs.

## Constraints

- Do not call WING/dLive, `send`, or any FX/insert setter.
- Do not promote FX apply or insert assignment into the supervised executor.
- Keep `queries_console=False`, `live_write=False`, `fx_apply_allowed=False`, and `insert_assignment_allowed=False`.
- Freshness must distinguish missing, unknown, stale, and fresh cache evidence.

## Expected Output

- `fx_inserts_snapshot.summary` includes `cache_freshness_status`.
- `fx_inserts_snapshot` includes top-level `cache_freshness` over normalized FX slot, FX parameter, and insert assignment fields.
- Read-surface inventory and fixture evidence mention FX/insert cache freshness while retaining physical proof, FX schemas, semantic names, and scoped diff blockers.
- Tests prove fresh, missing, and stale behavior without mixer calls.
