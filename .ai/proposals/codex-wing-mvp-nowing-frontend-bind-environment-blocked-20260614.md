# Proposal: no-WING frontend bind environment-blocked trace

## Short thesis

Classify only the known Playwright dev-server bind denial as
`environment_blocked`, keep the overall no-WING summary failed, and surface the
reason in JSON and markdown.

## What I understood

The no-WING quick gate is now the selected first action for the WING read-only
connection proof. In this sandbox, the frontend e2e subprocess cannot bind
`127.0.0.1:3000`, so the artifact should explain the environment failure
without implying a clean WING prestart gate.

## Proposed solution

- Add a narrow `frontend_e2e` stderr classifier for `listen EPERM` on
  `127.0.0.1:3000`.
- Mark that check as `environment_blocked`.
- Add `environment_blocked_checks` to the no-WING summary.
- Render an Environment Blocked Checks section in markdown.
- Cover the behavior with focused tests.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/reports/AUTOMIXER_WING_MVP_NOWING_FRONTEND_BIND_ENVIRONMENT_BLOCKED_20260614.md`
- `Docs/adr/wing-mvp-nowing-frontend-bind-environment-blocked-20260614.md`

## Alternatives considered

- Treat the check as passed in this sandbox: rejected because it would overstate
  frontend validation.
- Require `--skip-frontend`: rejected because the current first command is the
  default `--quick` path.
- Leave the raw failure only in logs: rejected because operators need a compact
  reason in the no-WING summary.

## Risks

The classifier could mask a real frontend failure if it matched too broadly.
The match is intentionally limited to `frontend_e2e` plus the concrete bind
denial text and address.

## Test plan

- Focused `tests/test_no_wing_release_check.py`.
- Runtime no-WING quick confirmation that `frontend_e2e` is reported as
  `environment_blocked`.
- Full standard suite before handoff.

## Where the other agent may disagree

Kimi may prefer keeping all frontend e2e infrastructure failures as plain
`failed`. The counterargument is that a structured environment-blocked reason is
more actionable while still fail-closing the no-WING gate.
