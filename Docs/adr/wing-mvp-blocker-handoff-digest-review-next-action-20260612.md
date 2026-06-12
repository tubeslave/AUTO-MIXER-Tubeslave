# ADR: WING MVP Blocker Handoff Digest Review Next Action

Date: 2026-06-12

## Context

The WING read-only handoff packet now contains a dedicated operator-sequence
step for reviewing `mvp_blocker_anchor` and `mvp_blocker_anchor_sha256` against
the current first open blocker. The no-WING validator exposes
`operator_sequence_has_mvp_blocker_anchor_digest_review: true`, but the WING MVP
blocker table is the live task source Project Director reads.

## Options considered

1. Keep the digest-review requirement only in handoff packet validation.
2. Document the requirement only in the runbook.
3. Add the requirement to the first blocker next action and regenerate progress.

## Decision

Use option 3. The first WING MVP blocker next action now requires
`operator_sequence_has_mvp_blocker_anchor_digest_review: true`.

## Why this won

Project Director reads the first blocker next action directly. Putting the
condition there keeps the selected action, generated progress, and no-WING
handoff validation aligned.

## Rejected alternatives

- Validator-only enforcement: too late for planning.
- Runbook-only prose: not the source Project Director selects from.

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
next-action wording and regenerates the latest progress report. No WING,
backend, or OSC behavior changes.
