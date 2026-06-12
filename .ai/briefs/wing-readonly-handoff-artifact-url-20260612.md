# Brief: WING read-only handoff artifact URL

## Context

The no-WING handoff command now carries the expected read-only backend URL. The
handoff packet artifact also stores `url`, but artifact schema review did not
reject a packet whose stored URL drifted from the expected backend URL.

## Task

Make handoff packet artifact review fail closed on URL drift.

## Acceptance

- Valid handoff artifact review reports the expected URL.
- A handoff artifact with a wrong URL is rejected.
- No WING/backend/OSC contact is introduced.
