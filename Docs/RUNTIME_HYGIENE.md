# Runtime Hygiene

## Supported backend entrypoints

- Repo-root import: `from backend.server import AutoMixerServer`
- Repo-root launch: `python -m backend.server`
- Electron dev launch: `frontend/public/electron.js` now starts the backend with `python -m backend.server` from the repository root

`backend/__init__.py` temporarily bootstraps the legacy sibling-import layout so repo-root imports work without `PYTHONPATH` overrides. This is an incremental compatibility layer, not a license to add more ad-hoc import hacks.

## Frontend modules currently status-only

The backend exposes read-only status endpoints for these tabs, but it does not expose live websocket control handlers yet. The frontend now treats them as disabled/status-only so it does not emit dead commands:

- Auto Panner: `start_auto_panner`, `stop_auto_panner`, `calculate_auto_panning`, `apply_auto_panning`
- Auto Reverb: `start_auto_reverb`, `stop_auto_reverb`, `calculate_auto_reverb`, `apply_auto_reverb`
- Auto Gate: `start_auto_gate`, `stop_auto_gate`, `configure_gate_channel`
- Auto Effects: `start_auto_effects`, `stop_auto_effects`
- Cross-Adaptive EQ: `start_cross_adaptive_eq`, `stop_cross_adaptive_eq`

These tabs should stay fail-closed until real backend handlers exist with explicit safety limits and test coverage.

## Active read-only surfaces on the primary backend path

These websocket surfaces are connected on the current `backend.server.AutoMixerServer` path and can be treated as supported without granting write authority:

- Metering and capture: `get_audio_capture_status`, `get_channel_meters`, `scan_audio_devices`, `select_audio_device`
- Mixer/discovery state: `get_state`, `scan_mixers`, `probe_wing_level_telemetry`, `start_wing_telemetry_capture`, `stop_wing_telemetry_capture`
- Safety state: `get_manual_write_supervision`, `get_manual_write_session`, `get_ready_for_live`, `get_supervised_pilot_status`, `get_freeze_status`

`get_freeze_status` only reports `auto_fader_v2` fields when an `auto_fader_v2_controller` is actually attached. The primary server bootstrap does not currently attach that controller, so those fields are optional rather than guaranteed runtime state.

## Disconnected backend surfaces now fail closed

These websocket commands remain in the dispatch table for compatibility, but the primary backend path does not attach the runtime they need. They now return explicit `status: unavailable` responses instead of crashing or silently doing nothing:

- Mixing-agent control: `get_agent_status`, `set_agent_mode`
- Mixing-agent approvals/history: `get_pending_actions`, `approve_action`, `approve_all_actions`, `dismiss_action`, `dismiss_all_actions`, `get_action_history`

## Backend-only commands that need UI planning or operator docs

These websocket commands are implemented server-side but are not part of the generic tab workflow. They should remain operator-only or receive dedicated UI before wider exposure:

- Supervised live-write safety: `get_supervised_pilot_status`, `get_manual_write_supervision`, `set_manual_write_supervision`, `start_manual_write_session`, `emergency_stop`
- Rollback and snapshot control: `create_snapshot`, `undo_snapshot_create`, `undo_restore_snapshot`, `rollback_last`, `rollback_all`
- Deployment readiness and gating: `get_ready_for_live`
- Telemetry and discovery: `scan_mixers`, `scan_audio_devices`, `select_audio_device`, `probe_wing_level_telemetry`, `start_wing_telemetry_capture`, `stop_wing_telemetry_capture`
- Freeze and automation supervision: `get_freeze_status`, `set_automation_frozen`, `set_channel_frozen`, `lock_channel`, `unlock_channel`

If any of these commands are surfaced in the UI later, they need explicit operator intent, clear dry-run/live state, and safe rollback affordances before they are allowed to touch the console.
