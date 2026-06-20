# M105 Project Director Post-M104 Intake Loop Guard

M105 records a direction guard, not new live authority. Project Director now
detects the recent post-M104 streak of WING read-only intake-hardening commits
and narrows further autonomous scope to
`external_validation_wait_or_named_gap_only`.

## What Changed

- Recent Project Director evidence now includes derived commit subjects.
- Consecutive `Guard WING readonly ...` subjects are counted as a WING
  read-only intake-hardening streak.
- After the M104 capture-time-order guard and at least ten consecutive WING
  read-only guard commits, the directive emits
  `pre_hardware_intake_hardening_loop_guard: true`.
- When the guard is active, further generic WING read-only intake guard
  milestones are forbidden unless a real WING artifact arrives, the operator
  explicitly requests the work, or a named unverified contract gap is present.

## Boundary

This is control-plane direction hardening only. It creates no WING/backend
contact, OSC/websocket transport, proposal apply, queue import, live write,
readiness closure, hardware evidence, Full Autonomous readiness or live
authority.

## Validation

Validation is recorded in `ops/autopilot/STATE.json` and
`ops/autopilot/RUN_LEDGER.jsonl`.

## Next Action

Wait for the operator-supervised WING read-only preflight using
`ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json`, or name a concrete
unverified contract gap before selecting more WING read-only intake work.
