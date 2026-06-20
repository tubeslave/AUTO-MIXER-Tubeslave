# M094 Project Director External-Validation Scope Guard

## Summary

M094 Project Director external-validation scope guard is now covered as
autopilot direction hardening. When the selected first non-closed blocker is
the WING read-only connection proof and the selected action requires real
WING/operator evidence, Project Director marks the cycle as an external
validation wait and narrows autonomous work to
`external_validation_handoff_or_control_plane_only`.

This addresses the risk that Goal mode keeps adding proposal-only context-gate
milestones while M001 is waiting for the real WING read-only artifact.

## Boundary

- No backend WebSocket was opened.
- No WING/OSC transport was opened.
- No live write was requested.
- No proposal apply path was enabled.
- No queue import path was enabled.
- No readiness closure was produced.
- No hardware evidence was produced.
- Full Autonomous Mixing Mode remains blocked.

## Validation

- Focused Project Director validation: recorded in `ops/autopilot/STATE.json`.
- M094 rail validation: recorded in `ops/autopilot/STATE.json`.
- Full WING rail: recorded in `ops/autopilot/STATE.json`.
- Clean staged full suite: recorded in `ops/autopilot/STATE.json`.
- Memory lint and JSON/JSONL validation: recorded in `ops/autopilot/STATE.json`.

## Operator Impact

Project Director still keeps Goal mode active, but while M001 waits for
external WING/operator validation it no longer treats broad offline
proposal-only gates as acceptable default progress. The allowed work is WING
read-only handoff/intake preservation and control-plane docs/state/rails.

## Remaining Blocker

M001 still waits for external WING/operator validation. Real live authority
remains unavailable until WING read-only proof, physical readback, rollback,
emergency-stop and supervised fader/gain evidence exist.
