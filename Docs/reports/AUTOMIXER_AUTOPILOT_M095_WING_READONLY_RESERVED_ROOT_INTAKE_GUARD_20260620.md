# M095 WING Read-Only Reserved-Root Intake Guard

## Summary

M095 WING read-only reserved-root intake guard is now covered as
external-validation intake hardening. Candidate WING read-only summaries from
`artifacts/no_wing_release_check/...` or `ops/autopilot/evidence/...` are
rejected before they can become accepted real WING read-only candidates.

This protects the M001 handoff from confusing no-WING/control-plane evidence
with the required real WING Rack artifact.

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

- Focused WING read-only evidence validation: recorded in
  `ops/autopilot/STATE.json`.
- M095 rail validation: recorded in `ops/autopilot/STATE.json`.
- Full WING rail: recorded in `ops/autopilot/STATE.json`.
- Clean staged full suite: recorded in `ops/autopilot/STATE.json`.
- Memory lint and JSON/JSONL validation: recorded in `ops/autopilot/STATE.json`.

## Operator Impact

Prestart no-WING summaries are still allowed as prestart evidence. They cannot
be accepted as the real WING read-only candidate. The future real artifact must
come through a non-reserved WING read-only artifact root and still needs manual
review before any readiness update.

## Remaining Blocker

M001 still waits for external WING/operator validation. Real live authority
remains unavailable until WING read-only proof, physical readback, rollback,
emergency-stop and supervised fader/gain evidence exist.
