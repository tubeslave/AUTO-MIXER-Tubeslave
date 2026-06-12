# ADR: WING MVP Blocker Handoff Anchor Next Action

Date: 2026-06-12

## Context

The no-WING handoff validator now requires handoff packets to include
`mvp_blocker_anchor` matching the current first open blocker row. The WING MVP
blocker table remains the live task source for Project Director. If the blocker
row does not mention the anchor requirement, future cycles can miss it even
though the validator enforces it later.

## Options considered

1. Keep the anchor requirement only in no-WING validator code.
2. Add the anchor requirement to documentation only.
3. Add it to the first blocker next action and regenerate progress output.

## Decision

Use option 3. The first blocker next action now requires
`mvp_blocker_anchor_matches_current_first_open: true` before supervised proof
planning.

## Why this won

Project Director reads the first blocker next action directly. Putting the
anchor requirement there makes the selected directive and generated progress
match the validator contract.

## Rejected alternatives

- Validator-only enforcement: too late for planning and easy to omit from
  operator handoff steps.
- Documentation-only note: weaker than updating the live blocker table.

## Implementation plan

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Extend tests for Project Director selected action and generated progress.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This change is documentation/tooling rail only. Rollback restores the previous
next-action wording and regenerates the latest progress report. No WING,
backend, or OSC behavior changes.
