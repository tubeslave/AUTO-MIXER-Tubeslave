# WING MVP Backend Prestart Command Text

## Context

The current first WING MVP blocker requires the failed busy-port no-WING summary
and no-WING handoff summary to show the backend prestart command exactly
matching `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

The machine contract already carries `backend_port_prestart_check_command` as an
argv list. Operators and downstream reports also need a stable exact-string
mirror so the command can be audited without interpreting JSON arrays or
Markdown rendering.

## Task

Add `backend_port_prestart_check_command_text` beside the existing argv-list
field across the read-only WING prestart artifact chain.

## Constraints

- Do not change the command.
- Do not remove or reinterpret `backend_port_prestart_check_command`.
- Do not start backend or WING transport.
- Do not grant live authority or readiness closure.
- Preserve the first open blocker: `WING read-only connection proof`.

## Test Plan

- Run focused WING preflight/review/proof-plan/no-WING tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run memory lint, `git diff --check`, and the standard full test suite.
