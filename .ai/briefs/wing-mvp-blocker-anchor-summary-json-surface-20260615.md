# Brief: WING MVP Blocker Anchor Summary JSON Surface

Date: 2026-06-15

## Task

Expose the WING MVP current first-blocker anchor fields as top-level no-WING
`summary.json` fields.

## Constraints

- Keep `WING read-only connection proof` as the first open blocker.
- Do not contact WING, query a console, start a backend/WING session, or widen
  live authority.
- Mirror the existing handoff-packet blocker anchor instead of changing
  readiness rules.
- Preserve packet-level readiness closure and live authority denial.

## Expected Evidence

- Top-level `summary.json` exposes `mvp_blocker_anchor`,
  `mvp_blocker_anchor_sha256`, and current-first-open match verdicts.
- Top-level `mvp_blocker_anchor.packet_does_not_close_blocker` remains `true`.
- Focused and standard tests pass.
