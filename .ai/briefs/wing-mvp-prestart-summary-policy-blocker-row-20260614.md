# WING MVP Prestart Summary Policy Blocker Row

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
Project Director uses the `## Current First Command` section plus the first
blocker table row as the selected action baseline.

The Current First Command already exposes
`backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes`. The first blocker table row must expose
the same marker so the baseline action cannot be weaker than the current
command.

## Task

Add the prestart summary connection-policy marker to the first WING blocker
table row and cover it with the focused WING rail test.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure from offline or candidate evidence.
- Keep the selected blocker as `WING read-only connection proof`.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused WING rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director JSON output and direction guard.
- Run `git diff --check` and memory lint.
- Run the standard full pytest command before commit.
