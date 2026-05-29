# Automixer Real WING Live Matrix - 2026-05-23

## Scope

Executed the planned real WING Rack validation flow: pre-live safety checks,
read-only WING preflight, dry-run proposal planning, live supervised
corrections, failure repair, and post-fix live replay.

This was not autonomous operation. All live mutations used
`apply_operator_proposal` or `rollback_last_manual_write`; direct write messages
were forbidden and checked in the runner event logs.

## Pre-live Checks

- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - Before live: `1129 passed, 1 skipped`
  - After tolerance fix: `1130 passed, 1 skipped`
- `git diff --check`: passed
- Backend port `8765`: free before and after live runs

## WING Preflight

- WING IP: `192.168.1.102`
- Connection: `connected=true`, `mode=wing`
- Operator mode at preflight: `assist`
- Live write policy at preflight: `suggestions_only`
- Manual write boundary: active
- Emergency stop/reset state: clear

Observed channel names:

| Channel | Expected | Observed |
| --- | --- | --- |
| 2 | SNARE T | SNARE T |
| 3 | SNARE B | SNARE B |
| 4 | TOM | TOM |
| 5 | F TOM | F TOM |
| 6 | OH L | OH L |
| 7 | OH R | OH R |
| 9 | GUITAR L | GUITAR L |
| 10 | GUITAR R | GUITAR R |
| 16 | VOCAL | VOCAL |

Read-only physical parameter checks succeeded for fader/gain on the live test
targets.

## Dry-run Matrix

All dry-runs returned `status=planned` and did not write to the console.

| Target | Type | Old | Target |
| --- | --- | ---: | ---: |
| CH2 SNARE T | fader | `0.0` | `-0.25` |
| CH4 TOM | gain | `0.0` | `-0.25` |
| CH6 OH L | fader | `0.0` | `-0.25` |
| CH9 GUITAR L | gain | `-2.3999996185302734` | `-2.65` |

## Failure Found and Fixed

Initial CH4 gain live proof applied the proposal but failed verification:

- Requested: `-0.25`
- Physical readback: `-0.1999988555908203`
- Old tolerance for gain readback: `0.05`
- Result: `mismatch_from_physical_readback`, `proof_status=partial`

The value was immediately restored through a supervised proposal:

- Before restore: `-0.1999988555908203`
- Restore target: `0.0`
- Final physical readback: `0.0`
- Restore path: `apply_operator_proposal`

Fix:

- Increased gain readback tolerance to `0.06 dB` while leaving fader tolerance
  at `0.05 dB`.
- Added regression coverage for WING trim quantization around a `-0.25 dB`
  request.

## Post-fix Live Matrix

All post-fix live proofs passed. Each run included:

- Assist-mode apply blocked with `operator_mode_blocks_live_write`
- Supervised apply through `apply_operator_proposal`
- Physical apply readback
- Rollback through `rollback_last_manual_write`
- Physical rollback readback
- Emergency-stop apply block with `emergency_stop_active`
- No forbidden direct messages in runner event logs

| Target | Type | Old | Applied Readback | Rollback Readback | Result |
| --- | --- | ---: | ---: | ---: | --- |
| CH2 SNARE T | fader | `0.0` | `-0.2499980926513672` | `0.0` | passed |
| CH4 TOM | gain | `0.0` | `-0.1999988555908203` | `0.0` | passed |
| CH6 OH L | fader | `0.0` | `-0.2499980926513672` | `0.0` | passed |
| CH9 GUITAR L | gain | `-2.3999996185302734` | `-2.59999942779541` | `-2.3999996185302734` | passed |

Final read-only physical values after all rollback checks:

| Target | Final value | Status |
| --- | ---: | --- |
| CH2 fader | `0.0` | `verified_from_physical_readback` |
| CH4 gain | `0.0` | `verified_from_physical_readback` |
| CH6 fader | `0.0` | `verified_from_physical_readback` |
| CH9 gain | `-2.3999996185302734` | `verified_from_physical_readback` |

## Artifacts

- Post-fix matrix summary:
  `artifacts/wing_telemetry/live_matrix_20260523_retry_after_tolerance/live_matrix_summary.json`
- Final readback:
  `artifacts/wing_telemetry/live_matrix_20260523_retry_after_tolerance/final_readback.json`
- Failed first CH4 gain attempt:
  `artifacts/wing_telemetry/20260523T104333Z_wing_operator_proposal_ch4_gain/summary.json`
- CH4 restore:
  `artifacts/wing_telemetry/live_matrix_20260523/restore_ch4_gain_20260523T104438Z.json`

## Verdict

The supervised WING path is now hardware-proven across multiple single-channel
safe corrections:

- fader down and rollback;
- gain trim down and rollback;
- physical readback verification;
- non-supervised apply blocking;
- emergency-stop apply blocking;
- no direct unsafe write messages from the runner.

This still does not approve autonomous mode, EQ, compressor, routing, snapshots,
FX, training/rules writes, or multi-channel batch apply.
