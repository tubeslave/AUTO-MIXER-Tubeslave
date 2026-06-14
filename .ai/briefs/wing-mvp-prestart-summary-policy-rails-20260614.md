# WING MVP Prestart Summary Policy Rails

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
Accepted-candidate prestart traces must bind to a passed no-WING release-check
summary before any supervised fader/gain proof planning.

Runtime validation already requires
`backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes`. The operator-facing rails must expose
the same marker beside the existing type, passed status, and no-contact fields.

## Task

Add the prestart summary connection-policy marker to the MVP stop-line,
Current First Command, autoloop prompt/README, and focused rail tests.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure from offline or candidate evidence.
- Keep the selected blocker as `WING read-only connection proof`.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused Project Director and WING rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director JSON output and direction guard.
- Run `git diff --check` and memory lint.
- Run the standard full pytest command before commit.
