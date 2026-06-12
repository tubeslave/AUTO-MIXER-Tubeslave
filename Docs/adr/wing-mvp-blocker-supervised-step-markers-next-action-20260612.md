# ADR: WING MVP Blocker Supervised Step Markers Next Action

## Context

The WING read-only handoff packet and no-WING validator require the final
supervised operator-sequence step to include:

- `accepted_decision_ready`
- `valid accepted-candidate prestart trace`
- `WING_SUPERVISED_FADER_GAIN_PROOF`

The first open MVP blocker selected action already required the supervised gate
to be present and final, but did not explicitly name those marker strings.

## Options considered

1. Keep the marker requirement only in validator/runbook surfaces.
2. Promote the markers into the first blocker next action.
3. Add a new runtime gate.

## Decision

Promote the three final-step markers into the first open WING MVP blocker next
action and generated progress.

## Why this won

Project Director uses the blocker table as the next-cycle source of truth. The
selected action should expose the same supervised proof transition conditions
that the handoff validator enforces.

## Rejected alternatives

Keeping the markers only in validator/runbook surfaces leaves the selected
action less explicit than the enforced contract. A new runtime gate would
duplicate existing structured handoff validation.

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
