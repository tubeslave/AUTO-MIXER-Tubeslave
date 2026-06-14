# ADR: WING MVP Prestart Summary Policy Rails

## Context

The accepted-candidate WING prestart trace must bind to a passed no-WING
release-check summary before supervised fader/gain proof planning. Runtime
validation requires that summary to report
`backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes`.

The current operator-facing rails already required the summary type, passed
status, and no-contact fields, but did not expose the connection-policy marker.

## Options considered

1. Keep the connection policy only in runtime validation.
2. Add the connection-policy marker to the MVP stop-line, Current First
   Command, and autoloop rails with focused tests.
3. Generate all prestart summary rail text directly from shared constants.

## Decision

Add `backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes` to the current WING MVP rails and cover
it with focused Project Director and WING rail tests.

## Why this won

It aligns operator-facing Goal-mode instructions with the existing runtime
contract while preserving the first blocker, no-live boundary, and manual
review requirement.

## Rejected alternatives

Runtime-only validation leaves a visibility gap in the exact surfaces agents and
operators read first. Constant-generated rail text is attractive but broader
than this blocker-first continuation slice.

## Implementation plan

- Add the connection-policy marker to `Docs/MVP_WING_SUPERVISED.md`.
- Add the marker to `.ai/project-layer/wing_mvp_blockers.md` Current First
  Command.
- Add the marker to autoloop prompt and README rails.
- Extend Project Director and WING rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk is limited to duplicated rail text. Rollback removes the marker and
focused assertions, then regenerates the latest progress report. No live WING
control, backend startup, console query, or readiness closure is introduced.
