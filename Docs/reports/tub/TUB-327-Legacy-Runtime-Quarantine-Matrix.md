# TUB-327 Legacy Runtime Quarantine Matrix

Date: 2026-05-17

Scope:
- duplicate or legacy runtime surfaces that can confuse the supervised pilot freeze window
- operator-visible surfaces first
- WING deployment safety over feature breadth

## Freeze-Window Baseline

The approved WING deployment path remains:

`AudioCapture -> AutoSoundcheckEngine -> ReplayWriteIntentAdapter -> supervised manual fader/gain write gate`

Source evidence:
- `Docs/RUNTIME_HYGIENE.md`
- `Docs/reports/tub/TUB-364-Analyzer-Control-Runtime-Consolidation.md`
- `backend/gain_fader_runtime.py`
- `tests/test_runtime_hygiene.py`
- `tests/test_gain_fader_runtime.py`
- `tests/test_auto_soundcheck_write_intents.py`

That means the freeze window should prefer:
- `python -m backend.server` for the primary backend surface
- `python3 start_soundcheck.py` or `AutoSoundcheckEngine` for dry-run analyzer work
- supervised manual writes as the only approved WING live-mutation path

Anything else should be clearly classified as one of:
- active and supported
- quarantined for the supervised pilot
- experimental or lab-only
- defer-until-after-pilot because it is not an operator-facing runtime surface

## Quarantine Matrix

| Surface family | Current entrypoint | Current reality | Freeze-window decision | Required follow-up |
|---|---|---|---|---|
| Voice control runtime | `start_voice_control` in `backend/server.py` | One live entrypoint exists, but it can instantiate `VoiceControlSherpa` with fallback to `VoiceControlV2`, while `VoiceControl` and `VoiceControlVosk` still exist as parallel implementations | Keep one supported voice entrypoint, but classify the runtime family as partially duplicated | Document Sherpa primary plus Whisper fallback, and mark `voice_control.py` / `voice_control_vosk.py` as non-deployment surfaces |
| Auto soundcheck UI orchestration | `frontend/src/components/AutoSoundcheckTab.js` -> `start_auto_soundcheck` | The UI still offers a full soundcheck cycle, but `backend/server.py` fans that cycle into reset, realtime correction, EQ apply, and auto-balance apply steps that are broader than the approved dry-run analyzer path | Quarantine on WING pilot surface | Hide or relabel the tab as lab-only on WING; keep `AutoSoundcheckEngine` as the supported analyzer path |
| Legacy auto-fader runtime | `frontend/src/components/AutoFaderTab.js` -> `start_realtime_fader`, `start_auto_balance`, `apply_auto_balance` | UI-visible and still wired to `AutoFaderController`, but the WING deployment boundary classifies it as a quarantined legacy runtime | Quarantine on WING pilot surface | Remove operator-facing start/apply affordances on WING, keep status and freeze reporting only |
| Experimental auto-fader v2 | `backend/auto_fader_v2/controller.py` and optional `get_freeze_status` fields | Test-covered and partially reused by helper services, but not instantiated by the primary server bootstrap | Keep experimental and unwired | Do not expose in pilot UI; only revisit after it can replace legacy gain/fader behind the same replay-intent contract |
| Mixing-agent runtime | `backend/handlers/agent_handlers.py` compatibility commands | Handler names still exist, but the server does not attach a `mixing_agent`; commands return explicit `status: unavailable` | Keep fail-closed compatibility only | Do not add UI; archive or remove only after the command surface is no longer needed for compatibility |
| Snapshot load/save/restore WebSocket surface | `load_snap`, `save_snap`, `restore_snapshot` | WebSocket handlers still exist, but WING deployment blocking quarantines the mutating paths | Keep quarantined | Keep out of pilot UI; if retained for operators, require explicit runbook-driven use outside the generic tab flow |
| Standalone snapshot/reset scripts | `backend/lab_only/legacy_snapshot_reset/*` | Direct-write operator scripts are archived behind a lab-only namespace instead of sitting at `backend/` root | Archived as lab-only | Keep out of pilot runbooks and require intentional `python3 -m backend.lab_only.legacy_snapshot_reset...` invocation |
| Status-only effect tabs | Auto Panner, Auto Reverb, Auto Gate, Auto Effects, Cross-Adaptive EQ | Already fail-closed in frontend and backend status endpoints | Keep as-is | No urgent work beyond preserving the disabled state |
| Duplicate analyzer/ML research modules | `backend/style_transfer.py` vs `backend/ml/style_transfer.py`, `backend/neural_mix_extractor.py` vs `backend/ml/neural_mix_extractor.py`, `backend/processing_graph.py` vs `backend/ml/processing_graph.py` | Real duplicate implementations exist, but they are not the current operator-facing deployment path | Defer until after pilot | Consolidate later with tests and ownership decisions; do not mix this cleanup into the pilot freeze |

## Freeze-Now Decisions

### 1. Keep supported

- `backend.server.AutoMixerServer` remains the primary backend runtime surface.
- `AutoSoundcheckEngine` remains the supported analyzer runtime on WING, but only in dry-run / replay-intent mode.
- Voice commands that resolve to supervised `fader` and `gain` writes can stay available because they already route through the manual write gate.
- Status-only modules stay disabled and should not be promoted during the freeze window.

### 2. Quarantine from operator-facing pilot flow

- `AutoSoundcheckTab` should not be treated as the deployment-approved analyzer path on WING because its cycle still orchestrates resets and apply steps from older controller generations.
- `AutoFaderTab` should not present itself as an active WING automation path because it still starts `AutoFaderController`, which the deployment runtime summary classifies as quarantined.
- Snapshot load/save/restore should remain out of the generic pilot flow even though compatibility handlers still exist.

### 3. Archive or relabel as lab-only

- `backend/lab_only/legacy_snapshot_reset/load_snap.py`
- `backend/lab_only/legacy_snapshot_reset/load_snap_v2.py`
- `backend/lab_only/legacy_snapshot_reset/load_snap_final.py`
- `backend/lab_only/legacy_snapshot_reset/find_and_load_snap.py`
- `backend/lab_only/legacy_snapshot_reset/scan_and_load_snap.py`
- `backend/lab_only/legacy_snapshot_reset/reset_all_channels.py`
- `backend/lab_only/legacy_snapshot_reset/reset_modules_trim_faders.py`
- `backend/lab_only/legacy_backups/server.py.bak`

These direct-write scripts are now parked behind the lab-only namespace because they look executable and authoritative, but they are not the approved supervised pilot runtime.

## Evidence Summary By Surface

### Voice

- `backend/server.py` starts `VoiceControlSherpa` first and falls back to `VoiceControlV2`.
- Voice execution is constrained to supervised `fader` and `gain` writes, while `load_snap`, `mute_channel`, EQ, and compressor commands are explicitly blocked.
- Separate `VoiceControl` and `VoiceControlVosk` implementations still exist in the repo, but they are not the chosen deployment entrypoint.

### Auto soundcheck and auto fader

- `backend/server.py` still contains the broader `start_auto_soundcheck` cycle, which calls `reset_all_functions_to_defaults`, `start_realtime_correction`, `apply_all_corrections`, `start_auto_balance`, and `apply_auto_balance`.
- `frontend/src/components/AutoSoundcheckTab.js` still exposes that cycle as a startable operator tab.
- `frontend/src/components/AutoFaderTab.js` still exposes `start_realtime_fader`.
- `backend/gain_fader_runtime.py` and `Docs/reports/tub/TUB-364-Analyzer-Control-Runtime-Consolidation.md` classify `AutoFaderController`, `LUFSGainStagingController`, and `SafeGainCalibrator` as legacy or quarantined on the WING deployment surface.

### Snapshot and reset surfaces

- `backend/handlers/snapshot_handlers.py` still registers `load_snap`, `save_snap`, and snapshot restore compatibility commands.
- `backend/server.py` blocks `restore_snapshot` on the WING deployment boundary, while `create_snapshot` remains available as a backup/export helper.
- Multiple standalone scripts in `backend/` still perform direct snapshot loads and whole-console reset style operations.

### Mixing-agent compatibility surface

- `backend/handlers/agent_handlers.py` keeps the message types alive, but returns `status: unavailable` when the runtime is absent.
- `Docs/RUNTIME_HYGIENE.md` already classifies those commands as disconnected and fail-closed.

## Recommended Safe Next Tasks

1. Remove or clearly label the WING-facing start controls in `AutoSoundcheckTab` and `AutoFaderTab`.
2. Add a small runtime badge in the operator UI using `gain_fader_runtime.selected_runtime` so the pilot surface states which runtime is active.
3. Done on 2026-05-17: moved the standalone snapshot/reset scripts under `backend/lab_only/legacy_snapshot_reset/`.
4. Add a voice-runtime note in the UI or runbook that states `VoiceControlSherpa` is primary and `VoiceControlV2` is fallback, while other voice backends are non-pilot.
5. After the pilot, consolidate duplicate analyzer research modules only with tests and clear ownership, not during the freeze window.
