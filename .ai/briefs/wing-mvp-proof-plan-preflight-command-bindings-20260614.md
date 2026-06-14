# WING MVP Proof Plan Preflight Command Bindings

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The handoff packet and runbook already bind the real read-only preflight run to
the passed no-WING prestart summary and the expected offline plan. The fallback
read-only preflight step emitted by the supervised proof plan also needs to
show those bindings so an operator does not follow a weaker command.

## Task

Require the supervised proof plan's `wing_readonly_preflight_required` step to
carry `--prestart-summary` and `--expected-plan`, and make
`tools/no_wing_release_check.py` validate and summarize those bindings.

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
