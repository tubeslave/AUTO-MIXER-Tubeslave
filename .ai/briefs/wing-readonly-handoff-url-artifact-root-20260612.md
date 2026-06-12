# Brief: WING read-only handoff URL and artifact root

## Context

The WING read-only plan and capture commands explicitly define the backend URL
and artifact root. The handoff command could rely on handoff tool defaults, but
the no-WING summary did not surface or validate those fields.

## Task

Make the handoff command explicitly carry the same URL and artifact root as the
read-only capture command.

## Acceptance

- The handoff command includes `--url ws://localhost:8765`.
- The handoff command includes `--artifact-root artifacts/wing_readonly_preflight`.
- no-WING evidence reports and validates both fields.
- No WING/backend/OSC contact is introduced.
