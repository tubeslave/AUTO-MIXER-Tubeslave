# TUB-335: WING Rack Telemetry Probe Notes

Date: 2026-05-14

## Purpose
Add a read-only telemetry probe path for WING firmware 3.0.5 observation mode and document expected/observed meter paths.

## Expected candidate paths (read-first)
1. `/$meters/subscribe`
2. `/$meters/ch/{channel}/in`
3. `/ch/{channel}/in`
4. `/ch/{channel}/in/$in`
5. `/ch/{channel}/main/1/lvl`
6. `/ch/{channel}/main/2/lvl`
7. `/ch/{channel}/main/3/lvl`
8. `/ch/{channel}/main/4/lvl`
9. `/ch/{channel}/send/1/lvl`
10. `/ch/{channel}/send/2/lvl`
11. `/ch/{channel}/send/3/lvl`

Compatibility fallback (non-level but useful telemetry signal verification):
- `/ch/{channel}/fdr`
- `/ch/{channel}/mute`
- `/ch/{channel}/name`

## Server API
- New websocket message: `probe_wing_level_telemetry`
  - Request fields:
    - `channels`: array of channels (default 1..8)
    - `timeout_sec`: wait time after probes (default 2.0)
    - `include_fallback`: include compatibility fallback addresses (default true)
  - Response message: `wing_level_telemetry`

## Failure modes (observability)
- `no_telemetry_response`: no values were returned for any candidate address.
- `exception`: probe execution failed.
- `no_non_meter_control_feedback`: only non-meter fallback values were seen (used for sanity checks).

If all candidates return empty and no exceptions, treat as unresolved firmware path and keep probing candidate list expanded.
