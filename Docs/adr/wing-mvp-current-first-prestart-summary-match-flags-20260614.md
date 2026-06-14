# ADR: WING MVP Current First Prestart Summary Match Flags

## Context

The WING read-only connection proof remains the first open MVP blocker. The
handoff packet validator now emits `*_matches_plan` verdicts for the no-WING
prestart summary contract. `Current First Command` is Project Director's
selected action source, so it should require those verdicts before any
supervised proof planning.

## Options considered

1. Leave the match flags only in validator output.
2. Add the flags to documentation only.
3. Add the flags to the Current First Command baseline and synchronized docs.

## Decision

Use option 3.

## Why this won

The baseline marker list is what tells Project Director whether the selected
action is complete enough to steer Goal-mode work. Raw summary values are useful,
but the match flags prove they are still bound to the plan artifact.

## Rejected alternatives

Validator-only coverage is too indirect for Goal-mode routing. Documentation-only
coverage would not fail the baseline when the selected action omits the flags.

## Implementation plan

- Add the prestart summary match flags to the baseline marker tuple.
- Add matching wording to `Current First Command`, MVP docs, and autoloop docs.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add focused tests for Project Director, rails, prompt/readme, and report
  visibility.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_project_director_agent.py tests/test_wing_mvp_rails.py --tb=short -x`
- Project Director JSON inspection.
- no-WING quick fail-closed check.
- Full standard pytest.

## Risks and rollback

This is a report-only rail change. It does not contact WING or grant live
authority. Roll back by removing the marker additions, synchronized docs, tests,
and regenerated progress report.
