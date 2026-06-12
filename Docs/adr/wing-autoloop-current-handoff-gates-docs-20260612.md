# ADR: WING Autoloop Current Handoff Gates Docs

## Context

The WING MVP blocker table now requires current handoff gates including
`mvp_blocker_anchor.packet_does_not_close_blocker: true` and final supervised
operator-step markers. Codex autoloop prompt and README guide future automated
cycles before task selection, so they must carry the same gate list.

## Options considered

1. Keep the new fields only in the blocker table.
2. Synchronize autoloop prompt and README.
3. Add a new runtime validation gate.

## Decision

Synchronize autoloop prompt and README with the current WING selected-action
handoff gates.

## Why this won

Future Codex cycles read the prompt and README before selecting work. Keeping
those surfaces aligned prevents the cycle contract from lagging behind the
blocker table and no-WING handoff validator.

## Rejected alternatives

Keeping the fields only in the blocker table leaves the prompt stale. A new
runtime gate would duplicate existing handoff validation.

## Implementation plan

- Update `tools/codex_autoloop/codex_cycle_prompt.md`.
- Update `tools/codex_autoloop/README.md`.
- Extend focused tests in `tests/test_project_director_agent.py`.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- Full pytest before commit.

## Risks and rollback

This change does not contact WING and does not grant live authority. Rollback is
to revert the prompt/README text, focused assertions, and report/index entry.
