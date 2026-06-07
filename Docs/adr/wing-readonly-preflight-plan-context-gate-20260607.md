# ADR: WING Read-Only Preflight Plan Context Gate

## Context

The supervised WING Rack MVP remains blocked on real read-only WING evidence.
The offline `--plan-only` artifact previews the exact request sequence before a
backend/WING session opens. That plan must also carry enough context to prevent
operator confusion: which MVP blocker it serves, that manual review is required,
which fail-closed conditions apply, and which review commands follow the real
capture.

## Options considered

1. Trust the runner tests and leave no-WING schema validation unchanged.
2. Validate only the request sequence and false live-authority flags.
3. Require the blocker/manual-review/fail-closed/review-command context in the
   plan artifact schema gate.

## Decision

Use option 3. The no-WING release check now rejects plan artifacts that omit or
drift from the required first-blocker context, and handoff packet artifacts fail
closed when their referenced offline plan artifact is missing or invalid.

## Why this won

The plan artifact is the operator-facing step immediately before the real WING
preflight. If it passes release checks without the manual-review and fail-closed
context, it can look more complete than it is. Validating the context keeps the
offline plan useful while preserving the real-artifact requirement.

## Rejected alternatives

- Runner-only tests do not protect generated artifacts consumed by release
  summaries.
- Request-sequence-only validation proves no network contact, but not the
  operator handoff around the first blocker.
- Duplicating expected command strings manually would drift from the command
  builders.

## Implementation plan

1. Add required fail-closed condition constants to no-WING release check.
2. Validate blocker/status, operator-required, manual-review boundary, fail-
   closed conditions, and post-run review commands.
3. Expose the fields through JSON and Markdown summaries.
4. Validate handoff packet artifacts against the referenced plan artifact.
5. Add positive and unsafe tests for the extended schema.
6. Document that this still does not close the WING read-only blocker.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py`
- Focused no-WING plan-artifact, handoff-artifact, and summary pytest.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-wing-plan-context --command-timeout-sec 300`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

This change does not contact WING, mutate console state, approve supervised
proof, update readiness, or close blockers. Rollback is to remove the added
schema fields, tests, and docs row; the real WING runbook remains unchanged.
