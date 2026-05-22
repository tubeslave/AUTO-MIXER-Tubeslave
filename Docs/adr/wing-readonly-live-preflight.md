# WING Read-Only Live Preflight

## Context

The operator reported that the physical WING Rack is connected and receiving
Ableton multitrack audio. A safe real-console path must prove observation first:
OSC state, audio meters, operator mode, and blocked write gates.

## Options Considered

1. Use an existing automation mode to start audio capture.
2. Add a read-only audio capture preview command.
3. Proceed directly to a supervised write smoke test.

## Decision

Add read-only audio capture preview handlers and keep live apply blocked by
default.

## Why This Won

The cockpit already had `get_channel_meters`, but no safe command to start the
underlying capture without activating correction logic. A preview command lets
the frontend and backend converge on the dashboard model before any console
write is attempted.

## Rejected Alternatives

- Automation mode as a meter bootstrap: too much behavior for observation.
- Immediate write test: premature while audio capture and proposal flow are
  still being validated.

## Implementation Plan

- Add `start_audio_capture_preview` and `stop_audio_capture_preview`.
- Normalize numeric audio device IDs before opening sounddevice streams.
- Add frontend service methods for those websocket commands.
- Stop discarded failed WING clients from leaving reconnect loops running.

## Test Plan

- `python3 -m py_compile backend/handlers/audio_handlers.py backend/server.py backend/osc/enhanced_osc_client.py tests/test_audio_handlers.py tests/test_server.py`
- `PYTHONPATH=backend pytest tests/test_audio_handlers.py tests/test_server.py tests/test_operator_product_state.py -q`
- `npm --prefix frontend run build`
- Manual read-only WING preflight against `192.168.1.102`.

## Risks And Rollback

The main risk is choosing the wrong audio device. Rollback is simple: stop audio
preview or select another device. No handler introduced by this decision writes
to the mixer.
