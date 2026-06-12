# ADR: WING MVP Blocker Supervised No-Live Authority Next Action

## Context

The WING read-only handoff packet and no-WING validator already require
`supervised_prestart_gate.grants_live_authority: false`. The current first open
MVP blocker selected action required the supervised gate to match
`expected_supervised_prestart_gate`, but did not name the no-live-authority
field directly.

## Options considered

1. Keep the requirement only in validator/runbook surfaces.
2. Promote the no-live-authority field into the first blocker next action.
3. Add a new runtime gate.

## Decision

Promote `supervised_prestart_gate.grants_live_authority: false` into the first
open WING MVP blocker next action and generated progress.

## Why this won

Project Director selects work from the blocker table. The selected action must
carry the same no-live authority boundary that the handoff validator and
runbook enforce.

## Rejected alternatives

Keeping the requirement only in validator/runbook surfaces leaves the next-cycle
directive less explicit than the enforced contract. A new runtime gate would be
duplicative because the structured gate equality already fails closed.

## Implementation plan

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused assertions in Project Director and MVP rail tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- Full pytest before commit.

## Risks and rollback

This change does not contact WING and does not grant live authority. Rollback is
to revert the blocker text, regenerated progress, and focused assertions.
