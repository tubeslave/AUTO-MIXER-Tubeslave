# ADR: WING MVP Test Execution Disallowed By Packet Next Action

## Context

The first non-closed WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff packet is an offline operator handoff artifact. It does not
contact a console, does not query a console, does not run tests, and does not
grant live authority.

The packet builder already exposes `executes_tests: false`, but the selected
action and runbook rails did not require that field before supervised proof
planning.

## Options considered

1. Leave `executes_tests: false` only in the packet JSON.
2. Promote `executes_tests: false` into the selected action, runbook, docs,
   autoloop prompt, and tests.
3. Generate every packet flag into every document from one source.

## Decision

Promote `executes_tests: false` into the operator-facing rails and tests.

## Why this won

This keeps the handoff honest: an offline packet can guide the operator
sequence, but it cannot be mistaken for executed validation. Real validation
must still come from the listed commands and real WING artifacts.

## Rejected alternatives

Leaving the field implicit is too easy to miss during handoff review. Generating
all docs from one source is larger than needed for this safety boundary.

## Implementation plan

- Add `executes_tests: false` to MVP stop rules and current next action.
- Add it to the read-only preflight handoff validation checklist.
- Add it to autoloop README and prompt checks.
- Add regression tests for Project Director, runbook, progress, and docs.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Risk: selected-action text grows further and becomes harder to maintain.
Rollback: remove the `executes_tests: false` rail additions and their tests.
This rollback would not change live WING behavior because no runtime live-write
authority is added here.
