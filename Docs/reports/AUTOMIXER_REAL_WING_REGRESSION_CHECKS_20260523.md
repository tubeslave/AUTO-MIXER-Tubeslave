# Automixer Real WING Regression Checks - 2026-05-23

## Scope

Executed the next real WING Rack validation pass after the earlier supervised
fader/gain proof. This pass covered repo baseline checks, frontend/Product
Layer checks, read-only WING preflight, dry-run planning, and a wider live
supervised regression matrix.

This was not autonomous operation. All live console changes went through
`apply_operator_proposal`; rollback went through `rollback_last_manual_write`.
No direct `set_fader`, `set_gain`, routing, snapshot, training, or rule write
messages were used.

## Repo And Frontend Baseline

- Backend tests:
  `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
  - Result: `1130 passed, 1 skipped`
- Frontend TypeScript:
  `npx tsc --noEmit`
  - Result: passed
- Frontend unit tests:
  `npm --prefix frontend test -- --watch=false`
  - Result: `13 passed`
- Frontend e2e:
  `npm --prefix frontend run test:e2e`
  - Result: `8 passed`
- Frontend production build:
  `npm --prefix frontend run build`
  - Result: passed
- Diff whitespace:
  `git diff --check`
  - Result: passed before live execution

## WING Read-only Preflight

- WING IP: `192.168.1.102`
- Connection: `connected=true`, `mode=wing`
- Operator mode: `assist`
- Live write policy: `suggestions_only`
- Manual write boundary: active
- Required live write mode: `supervised`
- Emergency stop before live matrix: clear
- Product Layer payloads checked:
  - `dashboard_snapshot`
  - `channel_inventory`
  - `groups_buses_snapshot`
  - `agent_communications_snapshot`
  - `connection_topology`

Read-only physical parameter queries returned
`verified_from_physical_readback` for fader/gain targets.

Observed relevant channels:

| Channel | Observed name | Final checked parameter |
| --- | --- | --- |
| 2 | SNARE T | fader |
| 3 | SNARE B | gain |
| 4 | TOM | gain |
| 5 | F TOM | read-only inventory |
| 6 | OH L | fader |
| 7 | OH R | read-only inventory |
| 9 | GUITAR L | gain |
| 10 | GUITAR R | read-only inventory |
| 16 | VOCAL | fader |

## Dry-run Matrix

All dry-runs returned `status=planned` and did not write to the console.

| Target | Type | Old value | Target value | Result |
| --- | --- | ---: | ---: | --- |
| CH2 SNARE T | fader | `0.0` | `-0.25` | planned |
| CH3 SNARE B | gain | `-11.899999618530273` | `-12.15` | planned |
| CH4 TOM | gain | `0.0` | `-0.25` | planned |
| CH6 OH L | fader | `0.0` | `-0.25` | planned |
| CH9 GUITAR L | gain | `-2.3999996185302734` | `-2.65` | planned |
| CH16 VOCAL | fader | `0.0` | `-0.25` | planned |

Dry-run artifacts:

- `artifacts/wing_telemetry/live_regression_20260523_dry/`

## Live Supervised Regression Matrix

Each live run included:

- apply attempt blocked in `Assist` with `operator_mode_blocks_live_write`;
- mode changed to `Supervised`;
- proposal applied through `apply_operator_proposal`;
- physical apply readback verified;
- rollback applied through `rollback_last_manual_write`;
- physical rollback readback verified;
- emergency stop triggered after rollback;
- post-emergency apply blocked with `emergency_stop_active`;
- emergency latch reset only with explicit approved operator reset before the
  next case.

| Target | Type | Old value | Target value | Apply readback | Rollback readback | Result |
| --- | --- | ---: | ---: | --- | --- | --- |
| CH2 SNARE T | fader | `0.0` | `-0.25` | physical verified | physical verified | passed |
| CH3 SNARE B | gain | `-11.899999618530273` | `-12.15` | physical verified | physical verified | passed |
| CH4 TOM | gain | `0.0` | `-0.25` | physical verified | physical verified | passed |
| CH6 OH L | fader | `0.0` | `-0.25` | physical verified | physical verified | passed |
| CH9 GUITAR L | gain | `-2.3999996185302734` | `-2.65` | physical verified | physical verified | passed |
| CH16 VOCAL | fader | `0.0` | `-0.25` | physical verified | physical verified | passed |

Live artifacts:

- `artifacts/wing_telemetry/live_regression_20260523/20260523T125223Z_wing_operator_proposal_ch2_fader/`
- `artifacts/wing_telemetry/live_regression_20260523/20260523T125224Z_wing_operator_proposal_ch3_gain/`
- `artifacts/wing_telemetry/live_regression_20260523/20260523T125225Z_wing_operator_proposal_ch4_gain/`
- `artifacts/wing_telemetry/live_regression_20260523/20260523T125226Z_wing_operator_proposal_ch6_fader/`
- `artifacts/wing_telemetry/live_regression_20260523/20260523T125226Z_wing_operator_proposal_ch9_gain/`
- `artifacts/wing_telemetry/live_regression_20260523/20260523T125227Z_wing_operator_proposal_ch16_fader/`

## Final Safe State

After the live matrix:

- final emergency reset: `status=reset`, `success=true`;
- final operator mode: `assist`;
- final live write policy: `suggestions_only`;
- manual write supervision: `emergency_stopped=false`, `reset_required=false`;
- `ready_for_live=false` remains correct because full gain-staging and
  auto-fader runtime checks are still unavailable.

Final physical readback:

| Target | Final value | Status |
| --- | ---: | --- |
| CH2 fader | `0.0` | `verified_from_physical_readback` |
| CH3 gain | `-11.899999618530273` | `verified_from_physical_readback` |
| CH4 gain | `0.0` | `verified_from_physical_readback` |
| CH6 fader | `0.0` | `verified_from_physical_readback` |
| CH9 gain | `-2.3999996185302734` | `verified_from_physical_readback` |
| CH16 fader | `0.0` | `verified_from_physical_readback` |

## Verdict

The supervised WING path is regression-proven for multiple single-channel,
small, safe corrections across fader and gain targets:

- physical WING readback works;
- Product Layer proposal apply works;
- rollback works;
- non-supervised apply is blocked;
- emergency-stop blocking works;
- emergency reset requires explicit operator approval;
- the final console values were restored.

This still does not approve autonomous mode, EQ, compressor, phase, routing,
snapshots, FX, training/rules writes, or multi-channel batch apply.
