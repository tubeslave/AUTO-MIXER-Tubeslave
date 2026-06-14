# Brief: WING MVP Prestart Backend No-Authority Detail

## Task id

`wing-mvp-prestart-backend-noauthority-detail-20260614`

## Context

The first supervised WING MVP blocker remains `WING read-only connection proof`.
The read-only preflight runner already requires a passed no-WING prestart
summary before opening the backend WebSocket, but the accepted `backend_port_free`
detail must prove no live authority explicitly, not only by omission.

Recent validation also showed that frontend E2E can mutate
`frontend/test-results/.last-run.json` even when the no-WING gate fails on a
sandbox bind boundary, so release checks need to preserve that generated local
artifact.

## Goal

- Make the no-WING backend-port prestart result always publish explicit
  no-authority detail flags.
- Make `wing_readonly_preflight_runner.py` reject prestart summaries where those
  backend-port detail flags are missing or not `false`, before WebSocket open.
- Preserve `frontend/test-results/.last-run.json` across the frontend E2E
  release check so validation does not leave unrelated dirty state.
- Record the decision and validation.

## Non-goals

- Do not contact WING.
- Do not grant live authority.
- Do not close the WING read-only connection proof blocker from offline
  evidence.
