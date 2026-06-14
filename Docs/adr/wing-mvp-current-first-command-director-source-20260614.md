# ADR: WING MVP Current First Command Director Source

## Context

The WING MVP blocker table identifies the first open blocker and evidence
source. The same document also has a `## Current First Command` section for the
latest operator-facing directive. Project Director and the progress report were
still selecting action text from the first table row only.

## Options considered

1. Keep Project Director tied only to the first table row.
2. Prefer `## Current First Command` for selected action text while keeping the
   first table row as blocker, evidence, and baseline safety source.
3. Replace the markdown command text with generated structured metadata.

## Decision

Prefer `## Current First Command` for selected action text in Project Director
and the generated MVP progress report, while appending the first table row's
baseline action text when both sources differ.

## Why this won

It keeps Goal-mode routing aligned with the most current operator command while
preserving older safety constraints from the table row. It does not change
blocker selection, live authority, backend behavior, or WING transport behavior.

## Rejected alternatives

Keeping only the table row preserves a known drift risk. A generated structured
contract is attractive but too broad for this blocker-first continuation slice.

## Implementation plan

- Add a scoped markdown-section reader to `update_mvp_progress.py`.
- Use the extracted command plus table-row baseline in `build_progress_report`
  and Project Director.
- Expose the selected-action source in Project Director output.
- Extend focused rail tests around the command source and diagnostics markers.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk is limited to command-source parsing and duplicated operator-facing text.
Rollback removes the helper, restores table-row-only selection, and regenerates
the latest progress report. No live WING control or readiness closure is
introduced.
