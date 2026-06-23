# Brief: WING MVP Operator Sequence Summary JSON Surface

Date: 2026-06-15

## Task

Expose WING read-only handoff operator-sequence contract fields as top-level
no-WING `summary.json` fields instead of requiring machine consumers to parse
the nested handoff packet artifact.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Preserve offline packet evidence as non-authoritative for readiness closure.
- Keep the change limited to no-WING reporting, tests, and Council evidence.

## Expected Evidence

- Top-level `summary.json` fields for operator sequence count, SHA256 binding,
  expected SHA256 binding, declared match flags, and declared count flags.
- The projected operator-sequence surface remains offline and explicitly
  denies readiness closure and live authority.
- Focused and standard tests pass.
