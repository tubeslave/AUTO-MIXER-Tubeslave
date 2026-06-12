# ADR: WING MVP Blocker Backend Prestart Next Action

## Context

The supervised WING MVP is still waiting on real WING read-only connection
proof. The top-level MVP rail now requires the structured
`backend_port_prestart_gate` to match `expected_backend_port_prestart_gate`
before any supervised fader/gain proof. The live blocker table and generated
latest progress report must not lag behind that contract.

## Options considered

1. Keep the stricter rule only in `Docs/MVP_WING_SUPERVISED.md`.
2. Copy the rule into `.ai/project-layer/wing_mvp_blockers.md` and regenerate
   `Docs/reports/MVP_PROGRESS_LATEST.md`.
3. Close or advance the blocker based on the stronger offline gate.

## Decision

Use option 2.

## Why this won

Project Director and progress visibility use the blocker table as the current
execution rail. The table therefore needs the same first-gate wording as the
MVP doc, while keeping every status unchanged until real WING artifacts exist.

## Rejected alternatives

Option 1 leaves an automation-facing stale next action. Option 3 is unsafe:
offline documentation and validation do not prove read-only hardware
connection.

## Implementation plan

- Patch the first blocker next action.
- Patch supervised fader and gain rows to require the backend prestart gate
  before supervised proof.
- Regenerate the latest progress report.
- Lock the wording with rails tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: operators could read the stronger gate as live readiness. Rollback is to
revert the table/report wording; blocker statuses remain unchanged throughout.
