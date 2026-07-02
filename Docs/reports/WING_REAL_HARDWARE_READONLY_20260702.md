# WING Real Hardware Read-Only Capture - 2026-07-02

## Scope

- Console: Behringer WING Rack at `192.168.1.102`
- Firmware observed by backend log: `3.0.5-4-ga4ea833d:release`
- Device identity observed by backend log: `ATMOSPHERA-WNG`, `wing-rack`, serial `0100Q1V0604AAE`
- Mode: backend `assist`, WING `observation_only=true`
- Live authority: not granted
- Live writes: not sent

## Persisted Evidence

- Tmp artifact root: `/tmp/automixer_real_wing_verified_subset_20260702T_tmp_persist/wing_readonly_preflight`
- Tmp run id: `20260702T094851Z_wing_readonly_preflight`
- Summary path: `/tmp/automixer_real_wing_verified_subset_20260702T_tmp_persist/wing_readonly_preflight/20260702T094851Z_wing_readonly_preflight/summary.json`
- Runner status: `passed`
- Runner reason: `wing_readonly_preflight_passed`
- `websocket_opened`: `true`
- `write_request_sent`: `false`
- `proposal_request_sent`: `false`
- `read_only`: `true`
- `live_write`: `false`
- `live_authority_granted`: `false`
- `readiness_closure_allowed`: `false`
- `readiness_matrix_update_allowed`: `false`
- `operator_mode`: `assist`
- `connection_mode`: `wing`
- `physical_readback_count`: `15`
- `readback_count`: `15`
- `blockers`: `[]`

The tmp artifact is a real WING read-only capture, but the review contract does
not accept it as a readiness candidate because the artifact root is not the
canonical `artifacts/wing_readonly_preflight` root. Treat it as supporting
evidence only, not as readiness closure.

## Verified Channel 3 Readbacks

| Value type | OSC address | Value |
| --- | --- | --- |
| `fader` | `/ch/3/fdr` | `-144.0` |
| `gain` | `/ch/3/in/set/trim` | `0.0` |
| `pan` | `/ch/3/pan` | `0.0` |
| `eq_on` | `/ch/3/eq/on` | `0` |
| `eq_mix` | `/ch/3/eq/mix` | `100.0` |
| `eq_low_gain` | `/ch/3/eq/lg` | `0.0` |
| `eq_high_gain` | `/ch/3/eq/hg` | `0.0` |
| `dyn_on` | `/ch/3/dyn/on` | `0` |
| `dyn_threshold` | `/ch/3/dyn/thr` | `-10.0` |
| `gate_on` | `/ch/3/gate/on` | `0` |
| `gate_threshold` | `/ch/3/gate/thr` | `-40.0` |
| `gate_model` | `/ch/3/gate/mdl` | `GATE` |
| `gate_ratio` | `/ch/3/gate/ratio` | `1:3` |
| `gate_attack` | `/ch/3/gate/att` | `10.0` |
| `gate_release` | `/ch/3/gate/rel` | `199.40428161621094` |

## Safety Result

This run proves read-only OSC query/readback for the listed surfaces on channel
3. It does not prove write/apply, rollback, or readiness closure. DSP methods
may use these mappings as readback evidence, but apply-capable corrections still
require verified write mapping, post-write readback, rollback proof, and an
operator decision.
