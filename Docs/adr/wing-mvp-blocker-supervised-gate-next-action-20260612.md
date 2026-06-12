# ADR: WING MVP Blocker Supervised Gate Next Action

Date: 2026-06-12

## Context

The no-WING handoff validator checks that the final operator-sequence step
contains the supervised prestart gate and that `supervised_prestart_gate` exactly
matches `expected_supervised_prestart_gate`. The WING MVP blocker table remains
the live task source Project Director reads.

## Options considered

1. Keep supervised gate requirements only in no-WING validator code.
2. Keep the requirement only in the read-only preflight runbook.
3. Add the requirement to the first blocker next action and regenerate progress.

## Decision

Use option 3. The first WING MVP blocker next action now requires
`operator_sequence_supervised_prestart_gate_is_final: true` and exact
`supervised_prestart_gate` / `expected_supervised_prestart_gate` parity.

## Why this won

Project Director reads the first blocker next action directly. Putting these
requirements there keeps planning, generated progress, and no-WING validation
aligned before supervised proof planning.

## Rejected alternatives

- Validator-only enforcement: too late for autonomous task selection.
- Runbook-only prose: useful for operators but not the selected-action source.

## Implementation plan

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Update tests and documentation index.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This is a documentation/tooling rail only. Rollback restores the previous
next-action wording and regenerates latest progress. No WING, backend, or OSC
behavior changes.
