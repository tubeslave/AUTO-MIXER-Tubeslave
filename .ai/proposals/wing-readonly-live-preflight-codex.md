# Codex Proposal: WING Read-Only Live Preflight

## Short Thesis

Before any real write test, the product cockpit needs a safe observation path:
WING OSC state plus audio input meters, with write gates still blocked.

## What I Understood

- The frontend architecture is now centered on operator modes and agents.
- A real WING Rack is currently receiving Ableton multitrack audio.
- The backend could read WING OSC state when started from the current process,
  but audio capture was not started from the cockpit path.
- Existing `get_channel_meters` only reports data after `server.audio_capture`
  is already running.

## Proposed Solution

- Add `start_audio_capture_preview` and `stop_audio_capture_preview` websocket
  handlers.
- Keep the operation read-only: it starts `AudioCapture` only, with no
  automation controller and no mixer writes.
- Normalize numeric string device IDs so frontend/PyAudio IDs like `"1"` open
  as sounddevice index `1`.
- Stop failed WING initial connection attempts so discarded clients do not leave
  reconnect loops running.

## Likely Files To Touch

- `backend/handlers/audio_handlers.py`
- `backend/server.py`
- `backend/osc/enhanced_osc_client.py`
- `frontend/src/services/websocket.js`
- `tests/test_audio_handlers.py`
- `tests/test_server.py`

## Alternatives Considered

- Start an existing automation mode only to get audio meters. Rejected because it
  couples observation to correction logic.
- Jump directly to a supervised write smoke test. Rejected because audio meter
  visibility is a prerequisite for meaningful live validation.

## Risks

- Device IDs differ between audio APIs. Numeric normalization reduces this for
  the current PyAudio/sounddevice device list.
- macOS local-network permissions can differ by launcher. The WING backend path
  should be launched from an environment that can reach the local network.

## Test Plan

- Unit-test preview start/stop handler behavior.
- Unit-test failed WING connection cleanup.
- Run product-state regression tests.
- Run frontend build.
- Manually verify WING read-only OSC connection and WING audio meters.

## Where Another Agent May Disagree

Another agent may prefer adding a full UI control immediately. I recommend
keeping the first patch at the backend/service bridge and wiring richer UI after
the live preflight path is stable.
