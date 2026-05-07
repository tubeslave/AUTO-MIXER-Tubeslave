1. Short thesis

Build a single "operator console" surface first, not another forest of module tabs. The
best direction is a dark, responsive dashboard centered on connection status, selected
channels, safe automation state, rollback, and a compact correction preview. Advanced
module pages can remain behind a secondary route.

2. What I understood

The frontend should control the running program from MacBook, iPad, and iPhone. It must
connect to the mixer, select the audio interface, choose channels for processing, roll
back settings, visualize applied corrections, and expose the important safety/operation
buttons. It should not show every metric or every backend feature.

The repository already has React/Vite/Electron and WebSocket commands for most of this:
`connect_wing`, `connect_dlive`, `connect_mixing_station`, `get_audio_devices`,
`get_channel_meters`, `create_snapshot`, `undo_restore_snapshot`, `bypass_mixer`,
`emergency_stop_agent`, `get_pending_actions`, and `get_action_history`.

3. Proposed solution

Option A: Show Control Console

- One primary screen for real use during rehearsal/live.
- Top safety rail: backend, mixer, audio device, snapshot age, Live/Soundcheck mode.
- Main area:
  - left: mixer/audio connection and selected device
  - center: enabled channels with level, protection state, and processing toggles
  - right: correction preview, pending actions, rollback, apply, bypass, emergency stop
- Best default because it matches the real operator workflow: "am I connected, what
  channels are under automation, what is the system about to change, can I undo it?"

Option B: Soundcheck Flow

- Guided sequence: Connect -> Select Channels -> Learn -> Review -> Apply.
- Fewer simultaneous controls, larger step state, strong before/after correction view.
- Best for setup and volunteer-friendly operation, but slower for live firefighting.

Option C: Touch Remote

- iPhone/iPad-first remote view with large touch targets and bottom navigation.
- Keeps only critical controls: mode, enable channels, pending changes, rollback,
  bypass, emergency stop.
- Best as a companion mobile route; too constrained to be the only desktop UI.

4. Likely files to touch

- `frontend/src/App.js`: route the new primary control surface and keep legacy tabs as
  secondary/advanced views.
- `frontend/src/App.css`: shared shell, responsive layout, dark theme tokens.
- `frontend/src/components/ControlSurface.js`: recommended new primary component.
- `frontend/src/components/ControlSurface.css`: dense desktop/tablet/mobile styling.
- `frontend/src/services/websocket.js`: add convenience methods for `create_snapshot`,
  `undo_restore_snapshot`, `scan_mixers`, `auto_connect`, `scan_audio_devices`, and
  `select_audio_device` if missing.
- Focused tests near existing frontend Node tests if business logic is extracted.

5. Alternatives considered

- Keep the current tab-heavy UI and restyle it. This is cheap, but it keeps every module
  equally visible and does not solve the "important controls only" problem.
- Build a full mixer clone with faders, EQ curves, compressor graphs, and meters for
  every channel. This is familiar but high-risk: it invites accidental operation and
  duplicates the physical console.
- Make a pure mobile remote first. Nice for iPhone, but MacBook/iPad users need better
  overview and review-before-apply workflows.

6. Risks

- Rollback semantics must be explicit: a channel backup is not the same as loading a
  console scene/snapshot. UI copy should say "Backup/Restore selected channels" rather
  than imply full-scene safety.
- Bypass currently resets all 40 WING channels to 0 dB and disables processing, so it
  must stay behind confirmation and probably should be visually separated from ordinary
  stop/pause controls.
- A compact correction preview needs normalized data from different modules. If backend
  event shapes differ too much, implement a frontend adapter rather than changing DSP
  logic.
- Mobile Safari/iPad operation must be tested against actual WebSocket networking and
  local-network permissions.

7. Test plan

- Frontend: `cd frontend && npm test && npm run build`.
- Backend safety regression: `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`.
- Manual with virtual mixer: connect, select audio device, select channels, create backup,
  start/stop automation, approve/dismiss pending actions, restore backup.
- Manual responsive QA at desktop, iPad portrait/landscape, and iPhone widths.

8. Where the other agent may disagree

- Kimi may prefer Option B first because it reduces operator error during soundcheck.
  That is reasonable if the next milestone is rehearsal setup rather than live control.
- Kimi may argue for keeping current module tabs as the main UI to reduce implementation
  risk. I would keep them, but make them secondary so the operator's first screen remains
  focused on safe control.
