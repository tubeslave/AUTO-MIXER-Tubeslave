# M107 Project Director Intake Loop Lookback Floor

M107 fixes the remaining horizon gap in the M105/M106 direction guard. After
M106, the Project Director could find the WING read-only intake streak inside
recent history, but normal `lookback_commits=12` evidence collection would stop
seeing the full streak after another direction-guard commit landed above it.

## What Changed

- Project Director now records requested lookback, effective lookback and the
  intake-loop guard lookback floor in its directive.
- Normal/default Project Director calls raise evidence collection to the
  intake-loop guard lookback floor before analyzing recent commit subjects.
- Focused Project Director tests cover multiple direction-guard commits above
  the WING read-only streak.
- The M107 rail proves current Project Director output remains
  `external_validation_wait_or_named_gap_only` with requested
  `lookback_commits=12`.

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
