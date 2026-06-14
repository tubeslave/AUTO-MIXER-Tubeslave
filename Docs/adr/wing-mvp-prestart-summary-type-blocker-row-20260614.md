# ADR: WING MVP Prestart Summary Type Blocker Row

## Context

The WING MVP blocker table is still the baseline source for the first open
blocker. Project Director also uses the `## Current First Command` section, but
it appends the table row so older safety constraints remain visible.

The current command already requires the accepted-candidate prestart trace to
bind to a no-WING release check summary. The table row needs the same
`backend_port_prestart_summary_type: no_wing_release_check` marker so baseline
Goal-mode text does not remain type-ambiguous.

## Options considered

1. Leave the marker only in `## Current First Command`.
2. Add the marker to the first blocker table row and cover it with a parser
   test.
3. Replace the blocker table row with generated structured metadata.

## Decision

Add `backend_port_prestart_summary_type: no_wing_release_check` to the first
WING blocker table row and assert it in `tests/test_wing_mvp_rails.py`.

## Why this won

It keeps both Project Director action sources aligned without changing backend
behavior, WING transport behavior, readiness closure, or live authority.

## Rejected alternatives

Leaving the marker only in the current command preserves a baseline drift risk.
Structured metadata is cleaner but too broad for this continuation slice.

## Implementation plan

- Insert the type marker into the `WING read-only connection proof` row.
- Extend the WING MVP rail test that parses the row.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Record validation in a dated report and index it.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to editing a very long markdown table row. Rollback removes the
marker and parser assertion, then regenerates the latest progress report. No
live WING control, backend startup, console query, or readiness closure is
introduced.
