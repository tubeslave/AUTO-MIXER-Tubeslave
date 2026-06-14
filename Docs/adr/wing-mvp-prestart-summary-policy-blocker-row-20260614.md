# ADR: WING MVP Prestart Summary Policy Blocker Row

## Context

The WING read-only connection proof remains the first open MVP blocker. Project
Director uses the Current First Command plus the first blocker table row as the
selected action source. Recent rails exposed
`backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes` in the Current First Command and
autoloop surfaces, but the first blocker table row still needed the same
baseline marker.

## Options considered

1. Keep the marker only in the Current First Command.
2. Add the marker to the first blocker table row and test the parsed row.
3. Refactor blocker-row text generation around shared runtime constants.

## Decision

Add the connection-policy marker to the first WING blocker row and cover it in
`tests/test_wing_mvp_rails.py`.

## Why this won

It keeps the Project Director baseline action as strict as the current command
without touching runtime WING code or changing the selected blocker.

## Rejected alternatives

Current-command-only coverage leaves a weaker table baseline in the action
source. Generated row text would reduce duplication but is broader than this
blocker-first continuation step.

## Implementation plan

- Add `backend_port_prestart_summary_connection_policy:
  no_hardware_no_network_no_live_writes` to the first blocker row next action.
- Assert the parsed row contains the marker.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Record validation in a report and link it from `Docs/INDEX.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to duplicated markdown text. Rollback removes the marker and
row assertion, then regenerates the latest progress report. No live WING
control, backend startup, console query, readiness closure, or live authority
is introduced.
