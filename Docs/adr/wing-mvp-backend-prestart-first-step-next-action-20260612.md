# ADR: WING MVP Backend Prestart First-Step Next Action

## Context

The WING read-only handoff packet carries backend-port prestart fields proving
that the diagnostic prestart check is the first operator step. The runbook and
validator already require these fields, but the first blocker selected action
did not explicitly name them.

## Options considered

1. Keep the fields only in runbook/validator surfaces.
2. Promote the fields into selected action and autoloop docs.
3. Add a new runtime gate.

## Decision

Promote the backend-port prestart first-step fields into the first WING MVP
blocker selected action, generated progress, and autoloop docs.

## Why this won

The first open WING blocker is the autonomous cycle entrypoint. It must state
that the backend-port prestart check happens before any intended backend/WING
session, without granting live authority.

## Rejected alternatives

Keeping the fields only in the runbook leaves the selected action less specific
than the handoff validator. A new runtime gate would duplicate existing
validation.

## Implementation plan

- Update blocker table/current command.
- Regenerate MVP progress.
- Update autoloop prompt/README.
- Add focused assertions.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- Full pytest before commit.

## Risks and rollback

This change does not contact WING and does not grant live authority. Rollback is
to revert the text, regenerated progress, focused tests, and report/index entry.
