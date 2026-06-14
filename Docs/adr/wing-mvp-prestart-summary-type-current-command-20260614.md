# ADR: WING MVP Prestart Summary Type Current Command

## Context

The accepted-candidate WING read-only prestart trace must bind to a passed
no-WING prestart `summary.json` before supervised fader/gain proof planning.
The shared validation contract expects that bound summary to identify itself as
`no_wing_release_check`.

The current blocker command, generated progress report, and autoloop rails
already require the prestart summary path, SHA256, passed status, no-contact
fields, and passed backend-port prestart status. They also need to carry the
summary type marker explicitly so Goal-mode routing cannot blur the accepted
source of the prestart evidence.

## Options considered

1. Leave the type requirement only in shared Python validation constants.
2. Add the type marker to the current blocker command, progress report, and
   autoloop rails while keeping runtime behavior unchanged.
3. Replace the markdown command text with generated structured metadata.

## Decision

Add `backend_port_prestart_summary_type: no_wing_release_check` to the current
operator-facing WING blocker command and autoloop rails, and cover it with
focused Project Director/progress/autoloop tests.

## Why this won

It makes the accepted-candidate prestart evidence source explicit in the places
Goal-mode and operators read first, without changing WING transport behavior,
backend behavior, live authority, or readiness closure.

## Rejected alternatives

Keeping the type only in runtime validation preserves a visibility gap. A fully
generated structured command may be cleaner later, but it is broader than this
safe blocker-first continuation slice.

## Implementation plan

- Add the type marker to `.ai/project-layer/wing_mvp_blockers.md` Current First
  Command.
- Add the same marker to the autoloop README and Codex cycle prompt.
- Extend focused rail tests for Project Director, progress report, and autoloop
  docs.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Record validation in a dated report and index it.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to duplicated operator-facing rail text drifting again. Rollback
removes the marker from the command/docs/tests and regenerates the latest
progress report. No live WING control, console query, backend startup, or
readiness closure is introduced.
