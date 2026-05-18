# TUB-338: WING live telemetry capture and replay comparison prep

Date: 2026-05-14

## What this heartbeat changed

- The WING server connect path now applies `observation_only` before telemetry
  capture attaches, so the very first recorded session events are explicitly
  read-only.
- The repo now has an offline comparator at
  `backend/wing_telemetry_compare.py` that can summarize one capture or diff two
  captures without contacting the console.

## Safe operator flow

1. Start the backend with:
   - `AUTOMIXER_WING_OBSERVATION_MODE=true`
2. Connect through the WebSocket `connect_wing` message with:
   - `observation_only=true`
   - `record_telemetry=true`
   - a unique `telemetry_session_label`
3. Run `probe_wing_level_telemetry` for the channels that matter.
4. Stop capture with `stop_wing_telemetry_capture`.
5. Keep the returned `events_path` and `metadata_path`.

## Offline comparison commands

Summarize one capture:

```bash
python3 backend/wing_telemetry_compare.py artifacts/wing_telemetry/<session_dir> --pretty
```

Compare a live and replay capture:

```bash
python3 backend/wing_telemetry_compare.py \
  artifacts/wing_telemetry/<live_session_dir> \
  artifacts/wing_telemetry/<replay_session_dir> \
  --pretty
```

## What the comparator flags

- event-type deltas between the two sessions
- `console_observed` addresses only present on one side
- per-address observation count mismatches
- safety warnings when either session contains `write_sent`
- fallback-only captures that only saw control addresses like fader/mute/name

## Remaining blocker for full closure

This repo still does not contain a real WING Rack live capture artifact. The
issue can only be fully closed after an operator records at least one
observation-only live session and, ideally, one replay session using the same
comparison workflow.
