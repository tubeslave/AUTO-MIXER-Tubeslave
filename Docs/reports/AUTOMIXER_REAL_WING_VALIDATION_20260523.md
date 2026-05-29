# Automixer Real WING Validation - 2026-05-23

## Verdict

Status: **supervised hardware-proven for fader proposal path on CH3**.

This run proves live WING connectivity, real multichannel audio capture, and one
safe Product Layer fader apply/rollback path with physical readback.

This is **not** autonomous/live-ready. Gain proof remains blocked because the
current backend state still does not expose `/ch/3/in/set/trim`.

## Runtime Setup

- Backend restarted from current checkout after the previous long-running
  `backend/server.py` process from 2026-05-19 did not respond to
  `get_manual_write_supervision`.
- WING connected through backend WebSocket `ws://localhost:8765`.
- WING IP: `192.168.1.102`
- Connection mode: `wing`
- Client: `EnhancedOSCClient`
- Operator mode during proof: `supervised`
- Live write policy during proof: `supervised_approval_required`

## Focused Safety Tests

Passed before live proof:

```bash
PYTHONPATH=backend python -m pytest \
  tests/test_wing_operator_proposal_validation_runner.py \
  tests/test_gain_proof_dry_run_runner.py \
  tests/test_server.py::TestWingDeploymentWriteBoundary -q
# 73 passed
```

## Read-Only Preflight

Fresh backend safety handlers responded:

- `get_operator_mode`: OK
- `get_manual_write_supervision`: OK
- `get_ready_for_live`: OK
- `get_dashboard_snapshot`: OK
- `get_state`: OK

Manual write state after fresh backend/WING connect:

- `deployment_write_boundary_active: true`
- `approved_live_write_kinds: ["fader", "gain"]`
- `emergency_stopped: false`
- `reset_required: false`

Named WING channels observed:

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

## Audio Capture Proof

Audio device list showed WING as device id `1` with 48 input channels.

Read-only preview was started:

```json
{
  "device_id": 1,
  "num_channels": 48,
  "sample_rate": 48000,
  "source_type": "sounddevice",
  "running": true
}
```

Active audio channels were detected from the WING device. Sample from the run:

- CH1: LUFS `-25.5`, peak `-12.1 dB`
- CH8: LUFS `-26.1`, peak `-20.5 dB`
- CH9: LUFS `-20.9`, peak `-11.3 dB`
- CH10: LUFS `-19.5`, peak `-9.6 dB`
- CH11: LUFS `-23.8`, peak `-10.9 dB`
- CH12: LUFS `-25.4`, peak `-14.3 dB`
- CH16: LUFS `-35.7`, peak `-18.1 dB`

Final read-only status still showed audio capture running with
`source_type: sounddevice`.

## Fader Proof

Target:

- Channel: CH3
- Name: `SNARE B`
- Value type: `fader`
- Old value: `0.0 dB`
- Proposed value: `-0.25 dB`
- Rollback target: `0.0 dB`
- Write policy: `apply_operator_proposal_only`

Dry-run:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type fader \
  --delta-db -0.25 \
  --timeout 8 \
  --rollback-delay-sec 1
```

Dry-run result:

- Run id: `20260523T073514Z_wing_operator_proposal_ch3_fader`
- Status: `planned`
- Old value: `0.0`
- Target value: `-0.25`

Live proof:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type fader \
  --delta-db -0.25 \
  --timeout 8 \
  --rollback-delay-sec 1 \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop
```

Live proof result:

- Run id: `20260523T073521Z_wing_operator_proposal_ch3_fader`
- Status: `passed`
- Assist-mode apply blocked with reason `operator_mode_blocks_live_write`.
- Supervised apply used `apply_operator_proposal`.
- Pre-read source: `physical_readback`
- Apply readback: `verified_from_physical_readback`
- Apply readback value: `-0.2499980926513672`
- Rollback source: `rollback_last_manual_write`
- Rollback readback: `verified_from_physical_readback`
- Rollback readback value: `0.0`
- Emergency stop proof: later proposal apply blocked with reason
  `emergency_stop_active`
- Final CH3 fader value from backend state: `0.0`

Artifacts:

- `artifacts/wing_telemetry/20260523T073514Z_wing_operator_proposal_ch3_fader/summary.json`
- `artifacts/wing_telemetry/20260523T073521Z_wing_operator_proposal_ch3_fader/summary.json`
- `artifacts/wing_telemetry/20260523T073521Z_wing_operator_proposal_ch3_fader/events.jsonl`
- `artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl`

## Gain Proof Status

Gain was checked only in dry-run mode after the fader proof.

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type gain \
  --delta-db -0.25 \
  --timeout 8 \
  --rollback-delay-sec 1
```

Result:

- Run id: `20260523T073536Z_wing_operator_proposal_ch3_gain`
- Status: `blocked`
- Reason: `current_value_missing_from_backend_state`
- Missing address: `/ch/3/in/set/trim`
- Execute: `false`

No physical gain write was executed.

## Final Safety State

After the successful fader proof, emergency stop is active by design:

- `manual_write_supervision.emergency_stopped: true`
- `manual_write_supervision.reset_required: true`
- `ready_for_live.ready: false`
- Blocked reasons include:
  - `gain_staging_unavailable`
  - `auto_fader_unavailable`
  - `manual_write_emergency_stop_active`

This state is correct for the proof. Do not silently reset it.

## Next Actions

1. Keep the readiness wording narrow:
   **supervised hardware-proven for fader proposal path on CH3**.
2. Before any new live write, perform explicit operator-approved emergency reset.
3. Do not attempt gain proof until `/ch/<channel>/in/set/trim` can be physically
   read and the runner dry-run returns `planned`.
4. Continue improving the gain old-value/readback path without broadening to
   EQ, routing, snapshots, scenes, training/rules, or autonomous apply.
