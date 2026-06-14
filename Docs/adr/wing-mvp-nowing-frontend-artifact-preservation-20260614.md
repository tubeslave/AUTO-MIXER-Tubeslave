# ADR: WING MVP no-WING Frontend Artifact Preservation

## Context

The first WING MVP blocker remains open until real WING read-only proof exists.
Offline no-WING checks are still the preflight validation path for Goal-mode
work. The frontend e2e command can rewrite
`frontend/test-results/.last-run.json`, a tracked Playwright metadata file,
leaving validation runs with unrelated dirty state.

## Options considered

1. Leave the artifact dirty and rely on manual cleanup.
2. Change Playwright configuration to write all e2e output elsewhere.
3. Add a per-command artifact preservation hook to the no-WING release checker.

## Decision

Use option 3.

## Why this won

The no-WING release checker owns the validation cycle that can dirty the
checkout. A per-command preserve hook keeps the fix local to that cycle, avoids
changing normal frontend test behavior, and can be validated with narrow unit
tests.

## Rejected alternatives

- Option 1 adds avoidable commit-prep noise and can hide meaningful local
  changes.
- Option 2 changes frontend test output behavior globally when only no-WING
  release checks need checkout preservation.

## Implementation plan

- Add `preserve_paths` to `CommandCheck`.
- Snapshot preserved paths before subprocess execution.
- Restore existing preserved files and remove newly created preserved files
  after subprocess execution.
- Mark `frontend/test-results/.last-run.json` as preserved for
  `frontend_e2e`.
- Validate preserved paths remain inside the repo boundary.
- Add focused tests for restore behavior and repo-boundary validation.
- Add the report to the current WING evidence trail and docs index.

## Test plan

- Focused no-WING release-check tests.
- Focused Project Director and WING rail tests.
- Project Director JSON check.
- Direction guard JSON check.
- `tools/memory/memory_lint.py`.
- `git diff --check`.
- no-WING quick check.
- Full pytest.

## Risks and rollback

Rollback removes the preserve hook, tests, and evidence/report references. This
does not contact WING, mutate console state, grant live authority, or close the
current blocker.
