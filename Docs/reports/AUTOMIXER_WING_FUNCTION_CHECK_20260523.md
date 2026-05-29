# Automixer WING Function Check - 2026-05-23

## Verdict

Status: **core WING functions checked; supervised fader path passed; gain path blocked safely**.

This check verified real WING connection, channel state reads, WING audio input
meters, operator-mode gating, supervised fader apply/rollback, and emergency
stop blocking. It did not test EQ, routing, snapshots, scenes, training/rules,
or autonomous apply.

## Safety Tests

Passed before live interaction:

```bash
PYTHONPATH=backend python -m pytest \
  tests/test_wing_operator_proposal_validation_runner.py \
  tests/test_gain_proof_dry_run_runner.py \
  tests/test_server.py::TestWingDeploymentWriteBoundary -q
# 73 passed
```

## WING Connection And Read-Only State

- Backend WebSocket: `ws://localhost:8765`
- WING IP: `192.168.1.102`
- Connection mode: `wing`
- WING connected: `true`

Named channels read from WING:

- CH1 `KICK`
- CH2 `SNARE T`
- CH3 `SNARE B`
- CH4 `TOM`
- CH5 `F TOM`
- CH6 `OH L`
- CH7 `OH R`
- CH8 `BASS`
- CH9 `GUITAR L`
- CH10 `GUITAR R`
- CH11 `PLAYBACK L`
- CH12 `PLAYBACK R`
- CH13 `ACCORDION`
- CH14 `BACK L`
- CH15 `BACK R`
- CH16 `VOCAL`

## Audio Input Meters

Read-only audio preview used WING audio device:

- Device id: `1`
- Channels: `48`
- Sample rate: `48000`
- Source: `sounddevice`

Active meter sample:

- CH1 peak `-30.5 dB`, LUFS `-26.8`
- CH2 peak `-52.0 dB`, LUFS `-52.1`
- CH3 peak `-46.7 dB`, LUFS `-48.8`
- CH4 peak `-46.6 dB`, LUFS `-52.1`
- CH5 peak `-36.8 dB`, LUFS `-44.1`
- CH6 peak `-40.1 dB`, LUFS `-44.0`
- CH7 peak `-44.1 dB`, LUFS `-45.8`
- CH8 peak `-18.0 dB`, LUFS `-26.9`
- CH9 peak `-18.9 dB`, LUFS `-20.7`
- CH10 peak `-17.3 dB`, LUFS `-19.5`
- CH11 peak `-14.7 dB`, LUFS `-27.5`
- CH12 peak `-15.7 dB`, LUFS `-29.4`
- CH13 peak `-24.7 dB`, LUFS `-33.1`
- CH16 peak `-14.3 dB`, LUFS `-20.7`

## Operator Mode And Safety Gate

Initial read-only state:

- Mode: `assist`
- Live write policy: `suggestions_only`
- Manual write emergency stopped: `false`
- Deployment write boundary active: `true`

During proof, runner switched to `supervised` only for approved proposal apply.
Assist-mode apply was blocked with:

- `operator_mode_blocks_live_write`

## Supervised Fader Apply/Rollback

Target:

- Channel: CH3
- Name: `SNARE B`
- Value type: `fader`
- Old value: `0.0`
- Target value: `-0.25`
- Rollback target: `0.0`

Dry-run:

- Run id: `20260523T101712Z_wing_operator_proposal_ch3_fader`
- Status: `planned`

Live proof:

- Run id: `20260523T101718Z_wing_operator_proposal_ch3_fader`
- Status: `passed`
- Apply path: `apply_operator_proposal`
- Pre-read: `verified_from_physical_readback`
- Apply readback: `verified_from_physical_readback`
- Rollback readback: `verified_from_physical_readback`
- Final CH3 fader: `0.0`

Artifacts:

- `artifacts/wing_telemetry/20260523T101712Z_wing_operator_proposal_ch3_fader/summary.json`
- `artifacts/wing_telemetry/20260523T101718Z_wing_operator_proposal_ch3_fader/summary.json`
- `artifacts/wing_telemetry/20260523T101718Z_wing_operator_proposal_ch3_fader/events.jsonl`
- `artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl`

## Gain Path

Gain proof was checked only as dry-run:

- Run id: `20260523T101729Z_wing_operator_proposal_ch3_gain`
- Status: `blocked`
- Reason: `current_value_missing_from_backend_state`
- Missing address: `/ch/3/in/set/trim`
- Execute: `false`

No physical gain write was executed.

## Final Safety State

Emergency stop proof intentionally leaves the system blocked:

- `manual_write_supervision.emergency_stopped: true`
- `manual_write_supervision.reset_required: true`
- `ready_for_live.ready: false`
- Final CH3 fader: `0.0`

This is expected. Do not silently reset emergency stop before the next live
write; reset requires explicit operator approval.
