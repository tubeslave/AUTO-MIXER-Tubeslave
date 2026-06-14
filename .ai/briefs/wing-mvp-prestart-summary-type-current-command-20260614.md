# WING MVP Prestart Summary Type Current Command

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The accepted-candidate prestart trace must bind to a passed no-WING prestart
`summary.json` before any supervised fader/gain proof planning.

The code-level contract already names the expected summary type as
`no_wing_release_check`. The operator-facing Current First Command, Project
Director selected action, generated progress report, and autoloop docs must
surface that same type so Goal-mode cycles do not treat any passed-looking
prestart summary as sufficient.

## Task

Require the current WING blocker command and autoloop rails to expose
`backend_port_prestart_summary_type: no_wing_release_check` beside the existing
accepted-candidate prestart summary status and no-contact fields.

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
