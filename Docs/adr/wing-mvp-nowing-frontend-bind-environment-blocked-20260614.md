# ADR: WING MVP no-WING frontend bind environment-blocked trace

## Context

The current WING MVP first command runs
`tools/no_wing_release_check.py --quick` before any intended backend/WING
session. In this sandbox, the Playwright frontend e2e web server cannot bind
`127.0.0.1:3000` and fails with `listen EPERM`.

## Options considered

1. Leave the failure as a generic failed frontend check.
2. Treat the frontend e2e check as passed when bind is denied.
3. Classify the known bind-denied case as environment-blocked while keeping the
   no-WING summary failed.

## Decision

Use option 3.

## Why this won

It preserves fail-closed no-WING behavior while making the artifact clearer for
operators and future Codex cycles. The summary explains that frontend e2e did
not run because the local environment blocked the dev-server bind.

## Rejected alternatives

Generic failure hides the actionable reason in subprocess logs. Treating it as
passed overstates validation and could weaken the WING prestart gate.

## Implementation plan

- Detect only `frontend_e2e` failures whose stderr contains `listen EPERM` and
  `127.0.0.1:3000`.
- Mark the check as `environment_blocked`.
- Add `environment_blocked_checks` to JSON summary.
- Add an Environment Blocked Checks markdown section.
- Add focused tests.

## Test plan

- `tests/test_no_wing_release_check.py`
- Runtime `tools/no_wing_release_check.py --quick` in the sandbox.
- Full standard suite.

## Risks and rollback

Risk: the classifier could be too broad. Roll back by removing the classifier
and the summary field, returning frontend e2e bind denials to generic failed
checks.
