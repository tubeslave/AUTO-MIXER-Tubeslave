# TUB-337: WING Rack Telemetry Capture Runbook

Date: 2026-05-14

## Purpose

Capture the first real WING Rack observation-only telemetry timeline as a durable replay dataset.

## Safety constraints

- Use `observation_only=true`.
- Do not run standalone write-capable scripts during the same capture.
- Prefer backend WebSocket entrypoints over direct ad hoc `WingClient(...)` scripts.

## Capture options

### Option A: enable capture on connect

Send `connect_wing` with:

```json
{
  "type": "connect_wing",
  "ip": "192.168.1.102",
  "send_port": 2223,
  "receive_port": 2223,
  "observation_only": true,
  "record_telemetry": true,
  "telemetry_session_label": "wing_live_probe"
}
```

### Option B: start capture after connect

```json
{
  "type": "start_wing_telemetry_capture",
  "telemetry_session_label": "wing_live_probe"
}
```

Stop and flush:

```json
{
  "type": "stop_wing_telemetry_capture",
  "reason": "live_probe_complete"
}
```

## Probe step

After connect, run:

```json
{
  "type": "probe_wing_level_telemetry",
  "channels": [1, 2, 3, 4],
  "timeout_sec": 3.0,
  "include_fallback": true
}
```

## Artifact layout

Default directory:

- `artifacts/wing_telemetry/`

Per session:

- `metadata.json`
- `events.jsonl`

## Event model

The JSONL timeline distinguishes:

- `query_sent`
- `write_sent`
- `blocked_write`
- `console_observed`
- connection lifecycle events

This separation is required for live-vs-replay diffing because local cache updates are not equivalent to console-observed state.

## Expected output

The backend returns `telemetry_capture` metadata in:

- `connection_status`
- `wing_level_telemetry`
- `wing_telemetry_capture_status`

Use the returned `events_path` and `metadata_path` as the durable dataset references for later replay comparison.
