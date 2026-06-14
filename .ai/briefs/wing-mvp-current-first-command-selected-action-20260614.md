# Brief: WING MVP Current First Command Selected Action

## Task

Keep Goal-mode Project Director focused on the first WING MVP blocker without
duplicating the full blocker-row baseline when `## Current First Command`
already carries the required safety and proof markers.

## Context

The first open blocker is still `WING read-only connection proof`. The previous
sync made `## Current First Command` self-contained enough to drive the next
operator step, but the progress tool still reported
`Current First Command + blocker table baseline` because the two strings are not
identical.

## Constraints

- Do not close readiness from offline evidence.
- Do not start the backend, contact WING, query a console, or write to a console.
- Preserve the no-WING prestart gate, listener-template denial fields,
  supervised-prestart gate, proof-plan summary fields, and accepted-candidate
  prestart-trace blockers.
- Keep the first blocker open until real WING artifacts are manually reviewed.

## Deliverable

Make Project Director use `Current First Command` directly only when that text
covers an explicit baseline marker set; otherwise retain the existing baseline
fallback.
