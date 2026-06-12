# WING Read-Only Runbook Value Types Match

## Summary

The WING read-only runbook examples now use `--value-types fader,gain,pan`, the
same read-only value-type set enforced by the runner and no-WING prestart gate.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No live authority granted.
- The command remains read-only.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks`
