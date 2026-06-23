# Brief: WING MVP no-WING frontend bind environment-blocked trace

## Task

Make the no-WING quick gate distinguish a frontend e2e infrastructure block
from an ordinary frontend regression when the Playwright dev server cannot bind
`127.0.0.1:3000` in the current sandbox.

## Context

`PYTHONPATH=backend /opt/homebrew/bin/python3 tools/no_wing_release_check.py --quick`
currently completes the WING/no-WING safety artifacts but exits 1 because
`frontend_e2e` fails with `listen EPERM: operation not permitted
127.0.0.1:3000`.

## Constraints

- Do not grant WING readiness from an environment-blocked frontend check.
- Do not hide the failed frontend e2e command.
- Do not contact WING hardware or start/stop backend listeners.
- Keep the change local to the no-WING gate and tests.

## Desired outcome

The summary remains fail-closed, but exposes a structured
`environment_blocked_checks` list and a per-check detail explaining that the
local dev-server bind was denied by the environment.
