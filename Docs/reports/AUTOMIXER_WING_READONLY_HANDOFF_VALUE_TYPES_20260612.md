# WING Read-Only Handoff Value Types

## Summary

The WING read-only handoff packet generator now fails closed when custom
`--value-types` are empty, duplicated, or outside `fader,gain,pan`.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No live authority granted.
- The handoff packet remains offline/read-only.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py`
