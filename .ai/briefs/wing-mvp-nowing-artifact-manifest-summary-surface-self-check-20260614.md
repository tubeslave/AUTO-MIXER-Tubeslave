# Brief: WING MVP no-WING artifact manifest summary surface self-check

## Task

Make the no-WING manifest summary surface fail closed when the operator-readable
`summary.md` does not render required manifest contract lines.

## Context

The main no-WING summaries now expose the detached `artifact-manifest.json`
contract. The next hardening step is to verify that the operator-facing markdown
actually renders those fields, not merely that JSON contains them.

## Constraints

- Do not contact WING or backend services.
- Do not grant live authority or readiness closure.
- Keep the check offline and deterministic.
- Preserve `artifact-manifest.json` as the final hash authority for
  `summary.json` and `summary.md`.

## Acceptance

- `summary.json` reports required manifest summary field lines.
- `summary.json` reports missing manifest summary field lines.
- `summary.json` reports whether those field lines rendered.
- Missing operator-readable manifest lines fail the no-WING summary closed.
- Project Director baseline requires the new self-check markers.
