# Brief: WING MVP Read-Only Prestart Summary Binding

Date: 2026-06-13
Task id: wing-mvp-readonly-prestart-summary-binding-20260613

## Goal

Make the WING read-only preflight candidate prove it is bound to the actual
passed no-WING prestart `summary.json`, not only to declarative prestart field
names.

## Scope

- Require `--prestart-summary` for WING read-only preflight planning and real
  runs.
- Record prestart summary path, SHA256, status, policy, live-write state, and
  `backend_port_free` status in read-only artifacts.
- Reject candidate artifacts when the referenced prestart summary is missing,
  stale, unsafe, or not passed.
- Keep all behavior offline/read-only until the operator runs the real WING
  preflight.

## Safety Boundary

- No WING hardware contact in this task.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness row closed from this tooling change.
