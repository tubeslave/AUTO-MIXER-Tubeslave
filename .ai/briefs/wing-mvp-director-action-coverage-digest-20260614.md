# Brief: WING MVP Director Action Coverage Digest

## Task

Make Project Director output auditable when it selects `Current First Command`
directly as the next WING MVP action.

## Context

The first open blocker remains `WING read-only connection proof`. Project
Director can now use `Current First Command` without appending the blocker-row
baseline, but downstream Goal-mode handoff needs machine-readable evidence that
the baseline marker coverage passed and that the selected action text did not
drift.

## Constraints

- Do not start the backend, contact WING, query a console, write to a console,
  close readiness, or grant live authority.
- Preserve fallback behavior when baseline coverage does not pass.
- Keep work scoped to the first WING MVP blocker and autoloop handoff.

## Deliverable

Project Director should publish a selected-action digest and explicit
current-first-command baseline coverage verdict, including missing markers.
