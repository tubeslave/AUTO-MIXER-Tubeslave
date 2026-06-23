# Brief: WING MVP Handoff Required-False Detail Contract

## Task

Keep the required backend-port no-authority detail fields visible and enforced
across the WING read-only offline plan, no-WING handoff packet, and generated
release summary.

## Context

The active WING MVP blocker remains `WING read-only connection proof`. The
read-only runner already rejects stale prestart summaries that lack explicit
backend-port no-authority detail, but the no-WING plan and handoff validation
must also fail closed if those fields are missing, `true`, or not bound between
plan and packet.

## Scope

- Validate the shared required-false backend-port detail fields in the offline
  read-only plan artifact.
- Carry the same fields in the no-WING handoff packet.
- Compare packet fields against plan fields and expose match verdicts in the
  no-WING summary.
- Render the fields in operator-visible Markdown.
- Record validation.

## Non-goals

- Do not contact WING.
- Do not grant live authority.
- Do not close the first WING blocker from offline evidence.
