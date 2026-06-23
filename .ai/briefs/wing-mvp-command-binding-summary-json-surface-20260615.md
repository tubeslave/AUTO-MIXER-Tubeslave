# Brief: WING MVP Command Binding Summary JSON Surface

Date: 2026-06-15

## Task

Expose WING read-only preview, capture, review, and decision-review command
bindings as top-level no-WING `summary.json` fields instead of requiring
machine consumers to parse the nested handoff packet artifact.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Preserve offline evidence as non-authoritative for readiness closure.
- Keep the change limited to no-WING reporting, tests, and Council evidence.

## Expected Evidence

- Top-level `summary.json` fields for preview/capture prestart summary path
  matches, artifact-root matches, plan-output matches, and review/decision
  command bindings.
- The projected command binding surface remains offline and explicitly denies
  readiness closure and live authority.
- Focused and standard tests pass.
