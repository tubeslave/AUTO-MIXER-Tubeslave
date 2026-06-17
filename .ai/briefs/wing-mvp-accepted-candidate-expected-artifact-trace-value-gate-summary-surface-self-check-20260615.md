# Brief: WING MVP Accepted Candidate Expected Artifact Trace Value Gate Summary Surface Self-Check

## Task

Make the no-WING accepted-candidate expected artifact trace value-gate summary
surface self-checking between top-level `summary.json` fields and the nested
`next_wing_proof_plan` surface.

## Context

The value gate is now exposed in `summary.json` and `summary.md`. A future edit
could still leave the top-level operator surface stale while the nested
`next_wing_proof_plan` values are correct, or the other way around. The no-WING
handoff should fail closed if those surfaces drift.

## Constraints

- No WING connection, OSC transport, backend WebSocket session, or live writes.
- Do not treat no-WING artifacts as readiness closure.
- Preserve the current first blocker: `WING read-only connection proof`.
- Keep readiness closure and live authority denied on the new self-check.

## Done When

- The no-WING summary exposes required fields, mismatches, match verdict, and
  readiness/live-authority denial for the value-gate summary surface self-check.
- A mismatch between top-level and nested value-gate fields marks the no-WING
  summary failed.
- `summary.md` renders the self-check mismatch list and match verdict.
- Focused and full validation pass.
