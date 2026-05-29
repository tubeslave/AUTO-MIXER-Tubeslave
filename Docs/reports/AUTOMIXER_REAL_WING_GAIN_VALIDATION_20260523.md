# Automixer Real WING Gain Validation - 2026-05-23

## Scope

Validated one narrow supervised gain proposal path against the real WING Rack.
This was not autonomous operation and did not use direct frontend/live write
messages. The only live mutation path was `apply_operator_proposal`, followed by
`rollback_last_manual_write`.

## Target

- Mixer: WING Rack `192.168.1.102`
- Channel: `3`
- Channel name observed by backend: `SNARE B`
- Parameter: gain trim `/ch/3/in/set/trim`
- Change: `-0.25 dB`

## Preflight Dry Run

Command:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type gain \
  --delta-db -0.25 \
  --timeout 12 \
  --rollback-delay-sec 1
```

Result:

- Status: `planned`
- Old value: `-11.899999618530273`
- Old value source: `physical_readback`
- Old value status: `verified_from_physical_readback`
- Target value: `-12.15`
- No write was executed.

## Live Supervised Proof

Command:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel 3 \
  --value-type gain \
  --delta-db -0.25 \
  --timeout 12 \
  --rollback-delay-sec 1 \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop
```

Result:

- Run id: `20260523T103454Z_wing_operator_proposal_ch3_gain`
- Status: `passed`
- Reason: `proposal_apply_readback_rollback_complete`
- Write policy: `apply_operator_proposal_only`

Apply proof:

- Requested value: `-12.15`
- Effective value: `-12.15`
- Old value: `-11.899999618530273`
- Pre-read status: `verified_from_physical_readback`
- Apply readback status: `verified_from_physical_readback`
- Apply readback value: `-12.100000381469727`
- Proof status: `verified`

Rollback proof:

- Rollback requested value: `-11.899999618530273`
- Rollback status: `applied`
- Rollback readback status: `verified_from_physical_readback`
- Rollback readback value: `-11.899999618530273`
- Rollback proof status: `verified`

Safety proof:

- Assist-mode apply attempt was blocked with
  `operator_mode_blocks_live_write`.
- Post-emergency apply attempt was blocked with `emergency_stop_active`.
- Runner event log contained no forbidden direct messages:
  `set_fader`, `set_gain`, `route_output`, `load_snap`, `save_snap`.

## Artifacts

- Summary:
  `artifacts/wing_telemetry/20260523T103454Z_wing_operator_proposal_ch3_gain/summary.json`
- Events:
  `artifacts/wing_telemetry/20260523T103454Z_wing_operator_proposal_ch3_gain/events.jsonl`
- Manual write gate log:
  `artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl`

## Final Readback

After rollback, a separate read-only `query_mixer_parameter` call returned:

- Address: `/ch/3/in/set/trim`
- Value: `-11.899999618530273`
- Source: `physical_readback`
- Status: `verified_from_physical_readback`

## Verdict

The real WING supervised gain proposal path is hardware-proven for one safe
scenario on CH3 `SNARE B`: apply, physical readback, rollback, rollback physical
readback, blocked non-supervised apply, and emergency-stop blocked apply.

This does not prove autonomous mode, EQ, routing, snapshots, training/rules
writes, or broad multi-channel behavior.
