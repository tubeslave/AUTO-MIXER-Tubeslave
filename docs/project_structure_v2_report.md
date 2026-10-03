# Project structure v2 report

Branch: `chore/project-structure-v2`

## What was changed in this first pass

This pass intentionally changed documentation only. No runtime code was moved, deleted, or rewritten.

Created files:

- `docs/cleanup_inventory.md`
- `docs/architecture.md`
- `docs/live_pipeline.md`
- `docs/offline_pipeline.md`
- `docs/safety_model.md`
- `docs/decision_engine.md`
- `docs/osc_mapping_wing.md`
- `archive/README.md`

## Files moved

None in this pass.

Reason: the backend server currently has many direct root-level imports. Moving files without a complete import graph and local test run would risk breaking live startup.

## Imports fixed

None in this pass.

Reason: no code files were moved.

## Legacy files

No files were archived yet.

`archive/README.md` now defines the policy for future archival.

## Live pipeline modules identified

Confirmed or likely live modules:

- `backend/server.py`
- `backend/wing_client.py`
- `backend/wing_addresses.py`
- `backend/dlive_client.py`
- `backend/mixing_station_client.py`
- `backend/osc/enhanced_osc_client.py`
- `backend/audio_capture.py`
- `backend/audio_devices.py`
- `backend/auto_soundcheck_engine.py`
- `backend/observation_mixer.py`
- `backend/lufs_gain_staging.py`
- `backend/channel_recognizer.py`
- `backend/auto_eq.py`
- `backend/phase_alignment.py`
- `backend/system_measurement.py`
- `backend/auto_fader.py`
- `backend/auto_compressor.py`
- `backend/bleed_service.py`
- `backend/feedback_detector.py`
- `backend/backup_channels.py`
- `backend/restore_channels.py`
- `backend/handlers/`
- `backend/ws_transport.py`

## Offline-only modules identified

Confirmed or likely offline/research modules:

- MuQ-Eval director/offline tests
- candidate renderers
- offline mix experiments
- critic/reward evaluators
- training/study/discovery routines
- benchmark/evaluation scripts

These must not directly send OSC or own a live mixer client.

## Safety violations found

No direct safety violation was confirmed in this pass because full repository text search was not available through the current connector session.

A future local scan should run:

```bash
grep -R "send_message\|send_osc\|udp_client\|SimpleUDPClient\|pythonosc" -n backend mix_agent src || true
```

Any direct OSC usage outside mixer transport modules should be reviewed and either allowed by exception or converted to typed action -> safety controller -> mixer client.

## Tests run

No tests were run in this environment.

Reason: this pass used the GitHub connector to create documentation files. A local clone/test execution was not available from the execution environment.

## Tests to run locally

Run these from the repository root:

```bash
python -m compileall backend
python -m pytest
```

Optional import/boundary scans:

```bash
find . -type f | sort > docs/file_list.txt
grep -R "^from \|^import " -n backend mix_agent src > docs/import_scan.txt || true
grep -R "send_message\|send_osc\|udp_client\|SimpleUDPClient\|pythonosc" -n backend mix_agent src > docs/osc_bypass_scan.txt || true
```

## Recommended next branch

Next branch:

```text
refactor/wing-mixer-package
```

Goal:

1. Create `backend/mixers/wing/`.
2. Move `backend/wing_addresses.py` to `backend/mixers/wing/wing_addresses.py`.
3. Move `backend/wing_client.py` to `backend/mixers/wing/wing_client.py`.
4. Keep compatibility shims at old paths.
5. Update `backend/server.py` imports only after smoke tests pass.
6. Add tests proving old and new imports both work.

## Recommended commit policy

Use small commits:

1. docs/inventory only
2. create package directories
3. move Wing addresses + shim
4. move Wing client + shim
5. update imports
6. add boundary tests
7. run tests and document result
