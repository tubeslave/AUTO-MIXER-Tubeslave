# WING Read-Only Runbook Handoff Plan Output

## Summary

The optional WING read-only handoff command now explicitly includes
`--plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json`, matching
the expected plan used by the plan-only and capture commands.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No live authority granted.
- The handoff packet remains offline/read-only.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks`
