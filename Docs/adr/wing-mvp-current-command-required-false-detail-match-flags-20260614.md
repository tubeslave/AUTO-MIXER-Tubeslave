# ADR: WING MVP Current Command Required-False Detail Match Flags

## Context

The first open WING MVP blocker is still `WING read-only connection proof`.
Recent no-WING handoff validation binds backend-port no-authority fields between
the offline read-only plan and handoff packet. Project Director uses `Current
First Command` as the selected action when baseline coverage passes, so that
baseline must include the new packet-vs-plan verdicts.

## Options considered

1. Leave the match verdicts in validator output only.
2. Add the verdicts to docs without changing the baseline marker set.
3. Add the verdicts to Current First Command, baseline markers, synchronized
   docs, and tests.

## Decision

Use option 3.

## Why this won

The selected action is the routing contract for Goal-mode work. Requiring the
five `*_matches_plan: true` verdicts there prevents a no-WING handoff from
looking baseline-complete while packet and plan disagree on no-authority detail
fields.

## Rejected alternatives

Validator-only coverage is too indirect for Project Director routing. Docs-only
coverage would not fail baseline coverage if Current First Command loses the
markers.

## Implementation plan

- Add the five required-false match verdicts to
  `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Add the same markers to Current First Command, the first blocker row, MVP
  docs, runbook, and autoloop docs.
- Extend existing focused tests for selected action, prompt/readme, reports, and
  rails.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- Project Director JSON inspection.
- no-WING quick check.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- Standard full pytest.

## Risks and rollback

This is an offline rail/report change. It does not contact WING, stop
listeners, or grant live authority. Roll back by removing the five marker
requirements, synchronized docs/tests, council trail, and regenerated progress
report.
