# WING MVP Current First Command Director Source

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The blocker document now carries a dedicated `## Current First Command` section
with the most current operator-facing command text.

Project Director and the generated progress report still read the first
blocker table row as the selected action. That makes the current command section
easy to drift from the actual Goal-mode directive.

## Task

Make `tools/codex_autoloop/project_director_agent.py` and
`tools/codex_autoloop/update_mvp_progress.py` prefer the `## Current First
Command` section when present, while preserving the first open blocker table as
the blocker selector, evidence source, and baseline safety constraint source.

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
