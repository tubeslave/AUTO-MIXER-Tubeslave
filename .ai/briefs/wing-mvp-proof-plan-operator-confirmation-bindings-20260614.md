# WING MVP Proof Plan Operator Confirmation Bindings

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The read-only preflight runner requires `--operator-confirm-channels` to match
`--channels` and requires `--operator-confirm-session-ready` whenever physical
readback is requested. The supervised proof plan fallback preflight command
must not show a weaker operator path.

## Task

Bind the supervised proof plan's `wing_readonly_preflight_required` command to
the canonical read-only preflight channel, value-type, operator channel
confirmation, session-ready confirmation, expected plan, and prestart summary
contract. Make `tools/no_wing_release_check.py` validate and summarize those
bindings.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure from offline or candidate evidence.
- Keep the selected blocker as `WING read-only connection proof`.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused supervised proof-plan and no-WING release-check tests.
- Run focused WING/readiness rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run `py_compile`, memory lint, `git diff --check`, and the standard full
  pytest command before commit.
