# Brief: WING MVP Project Director Value Gate Self-Check Markdown Surface

## Task

Make no-WING validation fail closed when the operator Markdown summary does not
render Project Director selected-action value-gate self-check fields.

## Context

The no-WING JSON summary exposes Project Director selected-action value-gate
self-check marker coverage. Operators also use `summary.md` before the real
WING read-only preflight. If the Markdown surface drifts, the operator can miss
a stale selected-action gate even when the JSON still carries the fields.

## Constraints

- No WING connection, backend WebSocket session, OSC transport, or live writes.
- Keep Project Director audit/planning-only.
- Do not close readiness from offline no-WING artifacts.
- Preserve the first open blocker: `WING read-only connection proof`.

## Done When

- no-WING runtime checks the Project Director value-gate self-check Markdown
  lines.
- Missing lines append a dedicated failed check and set summary status to
  `failed`.
- `summary.json` and `summary.md` expose the Markdown-surface verdict and keep
  readiness/live authority denied.
- Focused and full validation pass.
