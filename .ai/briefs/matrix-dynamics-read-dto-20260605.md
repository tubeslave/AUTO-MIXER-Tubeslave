# Matrix Dynamics Read DTO - 2026-06-05

## Task

Extend the cached Product Layer matrix snapshot with matrix dynamics readback fields without querying or mutating the console.

## Context

- Full autonomous mixing needs matrix visibility before master/output optimization can be trusted.
- `build_matrix_snapshot` currently exposes cached matrix fader/mute/pan, EQ, and insert state.
- `read_surface_inventory.matrix_state` still names dynamics as a missing matrix-state gap.
- Matrix dynamics writes are not in the supervised live executor and must remain blocked.

## Constraints

- Do not call WING/dLive, `send`, or any dynamics setter.
- Do not create matrix compressor/gate proposals or queue actions.
- Keep `queries_console=False`, `live_write=False`, and `full_autonomous_ready=False`.
- Preserve coverage metadata so missing cached dynamics is visible instead of silently ignored.

## Expected Output

- `matrix_snapshot` rows include cached `dynamics` with enabled/model/threshold/ratio/timing/makeup/mix fields.
- Matrix read coverage has a `dynamics` section.
- Read-surface inventory and fixture evidence mention cached matrix dynamics while retaining physical proof blockers.
- Tests prove complete and missing cached dynamics behavior without mixer calls.
