# Brief: WING MVP Accepted Candidate Expected Artifact Trace Summary JSON Surface

## Task

Expose the accepted-candidate expected artifact trace for the next WING proof
plan in no-WING `summary.json` and `summary.md`.

## Context

The current first WING command requires accepted review source ids to carry a
canonical expected-artifact trace before supervised proof planning:
`expected_artifacts`, `expected_summary_artifact`, `expected_events_artifact`,
`summary_path_matches_expected_artifact: true`, and
`events_path_matches_expected_artifact: true`.

The existing no-WING summary exposes candidate contract validity and violation
maps, but not the concrete accepted-candidate expected artifact trace as a
top-level machine-readable surface.

## Constraints

- No WING connection, OSC transport, backend WebSocket session, or live writes.
- Do not fabricate expected WING artifacts when there is no accepted candidate.
- Keep readiness closure and live authority denied.
- Preserve the first blocker: `WING read-only connection proof`.

## Done When

- no-WING `summary.json` exposes the accepted-candidate expected artifact trace
  by source id.
- `summary.md` renders the same fields in the Next WING Proof Plan section.
- The surface reports empty maps and `complete: false` until a real accepted
  WING candidate exists.
- Focused and full validation pass.
