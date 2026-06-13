# ADR: WING MVP Handoff Expected Artifacts Next Action

## Context

The WING read-only handoff packet now exposes an `expected_artifacts` list and
the no-WING gate verifies that the list matches the named summary/events globs
and the expected plan artifact list. The first WING MVP blocker is still
directed by `.ai/project-layer/wing_mvp_blockers.md`, and Project Director reads
that row directly.

## Options considered

1. Leave expected-artifacts checks only in no-WING summary output.
2. Add the checks to the canonical selected action and operator-facing rails.
3. Redesign the selected-action parser to generate all handoff checks from
   structured packet metadata.

## Decision

Add the expected-artifacts checks to the canonical selected action, MVP docs,
runbook, autoloop docs, progress output, docs index, and rail tests.

## Why this won

It keeps the current Goal-mode task selector aligned with the latest no-WING
artifact drift gate. Future cycles must now preserve the proof that packet
artifacts, named summary/events globs, and plan artifacts point at the same
real-run outputs.

## Rejected alternatives

Keeping the checks only in no-WING output is too easy to miss during a future
Project Director cycle. Reworking the parser is broader than needed for this
rail-hardening step.

## Implementation plan

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Update MVP/runbook/autoloop docs and docs index.
- Add focused rail assertions.
- Record this decision and validation report.

## Test plan

- `PYTHONPATH=backend python3 tools/codex_autoloop/update_mvp_progress.py --blockers .ai/project-layer/wing_mvp_blockers.md --output Docs/reports/MVP_PROGRESS_LATEST.md`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py tests/test_codex_direction_guard.py --tb=short`
- `PYTHONPATH=backend python3 tools/codex_autoloop/project_director_agent.py --format json --print`
- `PYTHONPATH=backend python3 tools/codex_autoloop/codex_direction_guard.py --format json`
- `python3 tools/memory/memory_lint.py`
- `git diff --check`

## Risks and rollback

Risk is limited to duplicated operator-facing rail text. Rollback removes the
new selected-action/doc/test assertions and the report entries. No live WING
behavior, backend startup, or console authority is introduced.
