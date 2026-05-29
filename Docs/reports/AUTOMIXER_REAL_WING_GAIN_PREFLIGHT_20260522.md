# Automixer Real WING Gain Preflight - 2026-05-22

## Verdict

Status: **blocked before physical gain write**.

No physical WING write was executed in this pass. The approved runner was used
only in dry-run mode and stopped before any proposal apply.

The current live readiness wording remains unchanged:

**Supervised hardware-proven for one fader proposal path.**

Do not upgrade readiness to gain-proven or autonomous-ready from this run.

## Repo State

- Baseline commit: `bb41592 Add Automixer cockpit and supervised safety handoff`
- Branch state before preflight: clean worktree, branch ahead of origin by 1
  commit.
- Backend WebSocket detected on `ws://localhost:8765`, PID `23699`.

## Focused Safety Tests

Passed before hardware preflight:

```bash
PYTHONPATH=backend python -m pytest \
  tests/test_wing_operator_proposal_validation_runner.py \
  tests/test_gain_proof_dry_run_runner.py \
  tests/test_server.py::TestWingDeploymentWriteBoundary -q
# 73 passed
```

## Read-Only WING Preflight

Read-only WebSocket checks showed:

- WING connection: connected
- WING IP: `192.168.1.102`
- Client: `EnhancedOSCClient`
- Operator mode: `supervised`
- Live write policy: `supervised_approval_required`
- Approved live write kinds: `fader`, `gain`
- Dashboard safety state: `manual_write_supervision.emergency_stopped == true`
- `ready_for_live.ready == false`
- Current state contains channel fader values and names, but no cached gain trim
  keys:
  - `trim_key_count: 0`
  - `/ch/3/in/set/trim` missing

Named channels observed in current backend state:

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

## Approved Runner Dry-Run

Command:

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

- Run id: `20260522T170353Z_wing_operator_proposal_ch3_gain`
- Status: `blocked`
- Reason: `current_value_missing_from_backend_state`
- Address: `/ch/3/in/set/trim`
- Execute: `false`
- Write policy: `apply_operator_proposal_only`
- Forbidden messages recorded by runner:
  `set_fader`, `set_gain`, `route_output`, `load_snap`, `save_snap`

Artifacts:

- Summary:
  `artifacts/wing_telemetry/20260522T170353Z_wing_operator_proposal_ch3_gain/summary.json`
- Events:
  `artifacts/wing_telemetry/20260522T170353Z_wing_operator_proposal_ch3_gain/events.jsonl`

## Blockers Before Physical Gain Proof

1. Emergency stop is active.
   - This must not be silently reset.
   - Reset requires an explicit operator-approved action and audit event.
2. Gain trim old value is not present in backend state.
   - The runner cannot prove old/current value for `/ch/3/in/set/trim`.
   - Without old value, rollback target cannot be proven.
3. The running backend does not respond to the separate
   `get_manual_write_supervision` request even though dashboard safety exposes
   manual write state.
   - This suggests the active runtime should be refreshed before the next live
     proof window, or the live preflight should rely only on dashboard safety
     until the runtime is restarted.

## Next Required Actions

Before attempting `--execute` for gain:

1. Refresh or restart the backend runtime from the current committed code in a
   controlled way, without killing unrelated processes.
2. Confirm `get_manual_write_supervision` responds.
3. Perform explicit operator-approved emergency stop reset and verify audit
   event.
4. Populate or physically read `/ch/<channel>/in/set/trim` before proposal
   apply.
5. Re-run the same runner in dry-run mode and require status `planned`.
6. Only then run the physical command with:

```bash
python3 tools/wing_operator_proposal_validation_runner.py \
  --url ws://localhost:8765 \
  --channel <LOW_RISK_CHANNEL> \
  --value-type gain \
  --delta-db -0.25 \
  --timeout 8 \
  --rollback-delay-sec 1 \
  --execute \
  --prove-blocked-apply \
  --prove-emergency-stop
```

If apply readback, rollback readback, or blocked proof is missing, status must
remain `partial/blocked`.
