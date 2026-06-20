# M106 Project Director Post-M105 Intake Loop Persistence

M106 fixes a persistence gap in the M105 direction guard. After M105 landed, the
latest commit was no longer a WING read-only guard, so prefix-only streak
detection turned the guard off even though the M095-M104 WING read-only intake
streak was still present in recent history.

## What Changed

- Project Director now finds the longest consecutive WING read-only intake
  guard streak inside recent commit subjects.
- The focused Project Director test now includes a leading direction-guard
  commit above the WING read-only streak.
- The M106 rail proves current Project Director output remains
  `external_validation_wait_or_named_gap_only`.

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
