# Brief: WING MVP Progress Director Evidence Summary

## Task

Make `Docs/reports/MVP_PROGRESS_LATEST.md` expose the same current-first-command
coverage and selected evidence count that Project Director uses for the active
WING MVP blocker.

## Context

The active blocker remains `WING read-only connection proof`. Project Director
already computes whether `Current First Command` covers the no-WING handoff
baseline and how many selected evidence files are bound to the first blocker,
but the operator-facing progress report only showed the next action text.

## Scope

- Share evidence-path extraction between the progress tool and Project
  Director.
- Add baseline coverage required/passed, marker count, missing markers, and
  selected evidence file count to `MVP_PROGRESS_LATEST.md`.
- Add focused rail coverage for the new report fields.
- Regenerate the latest progress report.
- Record the decision and validation.

## Non-goals

- No WING/backend contact.
- No live write changes.
- No readiness closure.
- No change to the first blocker status.

## Test plan

- Focused WING rails and Project Director tests.
- Project Director and direction guard JSON checks.
- Memory lint, diff whitespace check, no-WING quick check, and full pytest.
