# Brief: WING MVP no-WING Frontend Artifact Preservation

## Task

Prevent no-WING release checks from leaving tracked frontend Playwright
artifacts dirty after the frontend e2e command runs.

## Context

The active blocker remains `WING read-only connection proof`. Goal-mode cycles
use `tools/no_wing_release_check.py --quick` and related no-WING release checks
as local validation before any real WING proof attempt. The frontend e2e command
can update `frontend/test-results/.last-run.json`, which is tracked in this
checkout and can leave unrelated dirty state after a validation run.

## Scope

- Let no-WING command checks declare repo-local artifact paths that must be
  preserved across subprocess execution.
- Preserve `frontend/test-results/.last-run.json` for the frontend e2e check.
- Validate preserved paths stay inside the Automixer checkout.
- Add focused tests for existing-file restore and created-file removal.
- Record the decision and validation.

## Non-goals

- No WING/backend contact.
- No live write changes.
- No frontend test behavior changes beyond restoring the tracked helper file.
- No readiness closure.
- No change to the first blocker status.

## Test plan

- Focused no-WING release-check tests.
- Focused WING rails and Project Director tests after evidence updates.
- Memory lint, diff whitespace check, no-WING quick check, and full pytest.
