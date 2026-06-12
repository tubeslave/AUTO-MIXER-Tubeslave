# ADR: WING MVP Blocker Anchor Does Not Close Next Action

## Context

The WING read-only handoff packet includes an `mvp_blocker_anchor` that points
to the current first open blocker. The packet is an offline operator handoff and
must not close the read-only connection proof blocker. The runbook already
requires `mvp_blocker_anchor.packet_does_not_close_blocker: true`.

## Options considered

1. Keep the field only in runbook/validator surfaces.
2. Promote the field into the first blocker next action.
3. Add a new runtime gate.

## Decision

Promote `mvp_blocker_anchor.packet_does_not_close_blocker: true` into the first
open WING MVP blocker next action and generated progress.

## Why this won

Project Director selects the next autonomous cycle from the blocker table. The
selected action should state that the offline handoff packet does not close the
blocker before any supervised proof planning.

## Rejected alternatives

Keeping the field only in the runbook leaves the selected action less explicit
than the runbook. A new runtime gate would duplicate existing handoff packet
validation.

## Implementation plan

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused Project Director and WING MVP rail assertions.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- Full pytest before commit.

## Risks and rollback

This change does not contact WING and does not grant live authority. Rollback is
to revert the blocker text, regenerated progress, and focused assertions.
