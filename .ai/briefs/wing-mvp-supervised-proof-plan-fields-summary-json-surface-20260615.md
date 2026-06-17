# Brief: WING MVP Supervised Proof Plan Fields Summary JSON Surface

Date: 2026-06-15

## Task

Expose WING read-only handoff required supervised proof-plan summary field
contracts as top-level no-WING `summary.json` fields instead of requiring
machine consumers to parse the nested handoff packet artifact.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Preserve offline packet evidence as non-authoritative for readiness closure.
- Keep the change limited to no-WING reporting, tests, and Council evidence.

## Expected Evidence

- Top-level `summary.json` fields for required and expected supervised
  proof-plan summary fields, declared expected fields, match verdicts, and the
  invalid-record inclusion marker.
- The projected proof-plan field surface remains offline and explicitly denies
  readiness closure and live authority.
- Focused and standard tests pass.
