# Brief: WING MVP Project Director Value Gate Self-Check Summary Surface

## Task

Expose Project Director selected-action value-gate self-check coverage in the
no-WING handoff summary.

## Context

Project Director selects `## Current First Command` as the operator-facing
direction for the first open WING MVP blocker. The command now includes
accepted-candidate expected artifact trace value-gate summary-surface markers,
but no-WING output did not expose a dedicated top-level verdict proving those
markers remained present in the selected action.

## Constraints

- No WING connection, backend WebSocket session, OSC transport, or live writes.
- Keep Project Director audit/planning-only.
- Do not close readiness from offline no-WING artifacts.
- Preserve the first open blocker: `WING read-only connection proof`.

## Done When

- no-WING `summary.json` exposes Project Director selected-action value-gate
  self-check required, required markers, missing markers, pass, and denial
  fields.
- Operator `summary.md` renders the same fields.
- Missing value-gate self-check markers fail the handoff packet validation.
- Focused and full validation pass.
