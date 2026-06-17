# Brief: WING MVP Backend Prestart Summary JSON Surface

Date: 2026-06-15

## Task

Expose the backend prestart summary binding as top-level no-WING `summary.json`
fields instead of requiring machine consumers to parse the nested WING handoff
packet artifact.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Preserve offline evidence as non-authoritative for readiness closure.
- Keep the change limited to no-WING reporting, tests, and Council evidence.

## Expected Evidence

- Top-level `summary.json` fields for backend prestart summary path, SHA256,
  type, status, connection policy, safety flags, and match verdicts.
- The projected prestart summary remains a no-WING/offline binding and does not
  grant live authority or readiness closure.
- Focused and standard tests pass.
