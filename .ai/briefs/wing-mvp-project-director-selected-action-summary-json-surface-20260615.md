# Brief: WING MVP Project Director Selected-Action Summary JSON Surface

Date: 2026-06-15

## Task

Expose Project Director selected-action digest, match flags, baseline coverage,
and manifest-sidecar checklist verdicts as top-level no-WING `summary.json`
fields instead of requiring consumers to parse the nested handoff packet
artifact.

## Constraints

- Keep the first open blocker as `WING read-only connection proof`.
- Do not contact WING or widen live authority.
- Preserve offline evidence as non-authoritative for readiness closure.
- Keep the change limited to no-WING reporting, tests, and Council evidence.

## Expected Evidence

- Top-level `summary.json` fields for Project Director selected action source,
  SHA, current-command match, baseline coverage, sidecar missing markers, and
  sidecar denial markers.
- Existing Markdown and detached manifest surfaces remain unchanged.
- Focused and standard tests pass.
