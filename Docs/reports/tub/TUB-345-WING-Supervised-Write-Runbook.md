# TUB-345: First real supervised WING write test runbook

Date: 2026-05-14

## Purpose

Run the first physical `set_fader` + `set_gain` supervised-write test on the real
Behringer WING Rack using the already-implemented backend safety gate.

## Preconditions

- Backend server is already running on `ws://localhost:8765`.
- The chosen channel is safe:
  - not routed to PA/audience;
  - operator can watch the WING surface and stop immediately if needed;
  - absolute fader/trim targets are intentionally small for that channel.
- No autonomous write-capable flows are running during the test.
- Operator understands this is a real console write test.

## Recommended harness

Use the scripted harness instead of hand-sending JSON:

```bash
python3 backend/tub345_supervised_write_runner.py \
  --mixer-ip 192.168.1.102 \
  --channel 32 \
  --fader-target-db -60.0 \
  --gain-target-db 0.5 \
  --confirm-real-write
```

Optional:

- `--cooldown-fader-target-db -60.5`
- `--session-label tub346_supervised_live_manual1`
- `--artifacts-dir artifacts/tub345_supervised_live`

The harness saves a local JSON summary + full request/response transcript under:

- `artifacts/tub345_supervised_live/<session_label>_summary.json`

## What the harness does

1. `connect_wing` with:
   - `observation_only=false`
   - `record_telemetry=true`
   - `telemetry_session_label=<session_label>`
2. Confirms `get_manual_write_supervision` starts disarmed.
3. Arms supervised writes with `set_manual_write_supervision`.
4. Sends a negative-control `set_fader` with `approved=false` and expects `approval_required`.
5. Sends the positive supervised `set_fader` with `approved=true`.
6. Repeats a fader write immediately and expects `cooldown`.
7. Waits for the configured cooldown and restores the original fader value through the same supervised path.
8. Sends the positive supervised `set_gain` with `approved=true`.
9. Waits for the configured cooldown and restores the original gain value through the same supervised path.
10. Sends `emergency_stop`.
11. Verifies the next write is blocked with `manual_path_not_armed`.
12. Stops telemetry capture and prints/saves the resulting paths.

The runner now fails fast if the selected fader or gain target already matches the
current console value, because that would not prove a real write/readback cycle.

## Expected evidence to keep for the issue

From the harness summary JSON:

- chosen safe channel;
- requested fader and gain targets;
- `manual_write_blocked` negative-control result;
- `manual_write_applied` fader result with readback;
- `manual_write_applied` fader rollback result with restored readback;
- `manual_write_applied` gain result with readback;
- `manual_write_applied` gain rollback result with restored readback;
- cooldown-block result;
- post-`emergency_stop` blocked result.

From backend telemetry capture:

- `events.jsonl`
- `metadata.json`

The harness summary records the returned `events_path` and `metadata_path` so
they can be attached to the final Paperclip comment/report.

## Manual fallback messages

If the harness cannot be used, these are the key WebSocket messages:

```json
{"type":"connect_wing","ip":"192.168.1.102","send_port":2223,"receive_port":2223,"observation_only":false,"record_telemetry":true,"telemetry_session_label":"tub346_supervised_live"}
{"type":"get_manual_write_supervision"}
{"type":"set_manual_write_supervision","armed":true,"reason":"TUB-346 supervised rack test"}
{"type":"set_fader","channel":32,"value":-60.0,"approved":false,"reason":"TUB-346 supervised rack test"}
{"type":"set_fader","channel":32,"value":-60.0,"approved":true,"approval_id":"TUB-346-fader-1","reason":"TUB-346 supervised rack test"}
{"type":"set_fader","channel":32,"value":"<old_fader_value>","approved":true,"approval_id":"TUB-346-fader-rollback","reason":"TUB-346 supervised rack test rollback"}
{"type":"set_gain","channel":32,"value":0.5,"approved":true,"approval_id":"TUB-346-gain-1","reason":"TUB-346 supervised rack test"}
{"type":"set_gain","channel":32,"value":"<old_gain_value>","approved":true,"approval_id":"TUB-346-gain-rollback","reason":"TUB-346 supervised rack test rollback"}
{"type":"emergency_stop"}
{"type":"stop_wing_telemetry_capture","reason":"tub345_sequence_complete"}
```

Use the harness unless there is a concrete reason not to. It reduces operator
error and persists the exact transcript automatically.
