# WING MVP Prestart Summary Type Blocker Row

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
Project Director uses `## Current First Command` for selected action text, but
it also appends the first blocker table row as the baseline safety source.

The current command already exposes
`backend_port_prestart_summary_type: no_wing_release_check`. The first blocker
table row must expose the same marker so the baseline source cannot drift into
a type-ambiguous accepted-candidate prestart requirement.

## Task

Add `backend_port_prestart_summary_type: no_wing_release_check` to the first
WING blocker table row's next action and cover it with a rail test.

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
