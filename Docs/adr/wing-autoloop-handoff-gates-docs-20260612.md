# ADR: WING Autoloop Handoff Gates Docs

Date: 2026-06-12

## Context

The WING MVP blocker next action now requires blocker-anchor digest review and
supervised prestart trace gates before supervised proof planning. Codex autoloop
prompt and README still documented only the older backend/supervised gate
parity requirements.

## Options considered

1. Leave autoloop docs unchanged and rely on Project Director selected action.
2. Mention only the new fields in reports.
3. Update autoloop prompt, README, and tests to carry the current gate fields.

## Decision

Use option 3. Autoloop prompt and README now require the blocker-anchor match,
anchor digest match, digest-review operator step, supervised gate presence and
finality, and accepted-candidate prestart trace gate before supervised proof.

## Why this won

The prompt is the first contract each non-interactive Codex cycle reads. Keeping
it current reduces the chance that an automated cycle plans from stale WING
handoff requirements.

## Rejected alternatives

- Project Director only: good for task selection, but not the whole autoloop
  operating contract.
- Report-only note: not read early enough in the cycle.

## Implementation plan

- Update `tools/codex_autoloop/codex_cycle_prompt.md`.
- Update `tools/codex_autoloop/README.md`.
- Extend tests for prompt/README field coverage.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This is documentation/test coverage only. Rollback restores the older prompt
and README wording. No WING, backend, or OSC behavior changes.
