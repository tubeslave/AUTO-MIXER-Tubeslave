# ADR: WING MVP Existing Backend Required By Packet Next Action

## Context

The first non-closed WING MVP blocker remains `WING read-only connection proof`.
The no-WING handoff packet is an offline operator handoff artifact. It does not
contact a console, query a console, run tests, execute WING commands, or grant
live authority.

The packet builder already exposes
`physical_preflight_requires_existing_backend_connection: true`, but the
selected action and runbook rails did not require that packet field before
supervised proof planning.

## Options considered

1. Leave the field only in the packet JSON.
2. Promote the field into the selected action, runbook, docs, autoloop prompt,
   and tests.
3. Generate every packet flag into every document from one source.

## Decision

Promote `physical_preflight_requires_existing_backend_connection: true` into the
operator-facing rails and tests.

## Why this won

This keeps the handoff honest: a physical read-only preflight must use the
intended existing backend/WING session. The offline handoff packet must not be
read as direct connect or disconnect authority.

## Rejected alternatives

Existing runbook prose is useful but not enough for Project Director selected
work. Generating all docs from one source is larger than needed for this safety
boundary.

## Implementation plan

- Add the field to MVP stop rules and current next action.
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
Rollback: remove the existing-backend rail additions and their tests. This
rollback would not change live WING behavior because no runtime live-write
authority is added here.
