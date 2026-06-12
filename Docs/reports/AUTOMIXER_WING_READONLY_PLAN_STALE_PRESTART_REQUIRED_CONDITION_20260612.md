# WING Read-Only Plan Stale Prestart Required Condition

## Summary

The no-WING release-check required fail-closed list now includes
`stale_backend_port_prestart_detail_contract`, matching the read-only runner and
backend evidence contract.

## Safety Boundary

- No WING hardware contact.
- No backend WebSocket opened.
- No OSC writes.
- No live authority granted.
- The change keeps the offline expected-plan gate fail-closed.

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "wing_preflight_plan_artifact"`
  - `4 passed, 194 deselected`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`
  - fail-closed only on `backend_port_free` because `127.0.0.1:8765` already
    has a local Python listener requiring operator confirmation
- no-WING quick probe with `--continue-on-failure`
  - remaining failed checks: `backend_port_free` only
- Adjacent WING/no-WING tests
  - `244 passed`
