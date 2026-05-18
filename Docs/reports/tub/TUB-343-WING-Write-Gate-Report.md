# TUB-343 WING Supervised Write Gate Report

## Scope completed in this change

- Added a central supervised-write gate in `backend/wing_client.py`.
- Defaulted WING writes to `disarmed`.
- Allowed only supervised channel `fader` and `gain` writes when armed.
- Required operator `reason` and approval metadata for supervised writes.
- Added readback verification and rollback attempt on mismatch.
- Latched the gate into `emergency_stopped` after readback failure.
- Blocked raw direct OSC writes with values unless they run inside the supervised path.

## Write paths reviewed

- WebSocket manual writes:
  - `set_fader` -> routed through shared supervised path.
  - `set_gain` -> routed through shared supervised path.
  - `set_eq` -> explicitly blocked in phase 1.
  - `set_compressor` -> explicitly blocked in phase 1.
- Voice writes:
  - `set_fader`, `set_gain`, `volume_up`, `volume_down` -> routed through shared supervised path and therefore require arm + approval.
  - `load_snap`, `mute_channel`, EQ and compressor voice commands -> explicitly blocked in phase 1.
- Direct `WingClient` calls:
  - Any write that reaches `send(address, value)` outside the supervised context is blocked centrally.
  - This covers legacy scripts and most standalone/headless paths that call WING setters directly.
- Headless auto engine:
  - `backend/handlers/soundcheck_handlers.py` now forces `auto_apply=False` for `mixer_type == "wing"`.

## Remaining risky paths to audit next

- Legacy automation modules that call WING writes repeatedly may now be safely blocked, but they are not yet refactored onto the supervised path.
- Non-server scripts under `backend/` still need a pass to decide which should stay blocked, which should become dry-run only, and which should be upgraded to use explicit supervised APIs.
- Some reset/apply helpers in `backend/server.py` still issue direct write intents; the new gate blocks them on WING, but they are not yet converted into user-facing blocked responses.

## Safety notes

- `ready_for_live.no_clipping` no longer reports a fake `true`.
- `config/automixer.yaml` merge markers were removed so live tests do not consume an invalid YAML file.

## Verification

- `python3 -m pytest -q tests/test_wing_client.py tests/test_server.py`
- Result: `51 passed`
