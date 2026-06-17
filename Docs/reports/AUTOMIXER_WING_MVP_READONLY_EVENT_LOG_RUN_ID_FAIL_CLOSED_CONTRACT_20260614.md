# WING MVP Read-Only Event Log Run-Id Fail-Closed Contract

## Summary

The WING read-only preflight plan and handoff contract now explicitly list
`event_log_run_id_mismatch` as a fail-closed condition.

## Changed

- `backend/wing_readonly_preflight_contract.py` adds the marker to the shared
  `WING_READONLY_PREFLIGHT_PLAN_FAIL_CLOSED_CONDITIONS` tuple.
- The runner, handoff packet, and no-WING release check consume the same shared
  tuple, so the marker appears in operator-facing plan and packet surfaces.
- Focused tests assert the marker is visible in runner output, no-WING plan
  detail, and handoff markdown.

## Safety Boundary

This is a read-only artifact-contract change. It does not contact WING, mutate
console state, grant live authority, close readiness, or authorize supervised
fader/gain proof.

## Remaining Blocker

The WING MVP still waits for a fresh real `wing_readonly_preflight.v1` capture
and operator review before supervised proof planning can advance.
