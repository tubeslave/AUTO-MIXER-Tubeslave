# ADR: WING MVP Runbook Accepted-Decision Artifact-Root Shape

## Context

The accepted review decision schema requires `artifact_root`, and generated
pending decision templates include it. The WING read-only runbook minimum JSON
example omitted the field.

## Options Considered

1. Leave the runbook example shorter and rely on the generated template.
2. Make the runbook example match the required decision schema.

## Decision

Use option 2. The minimum accepted-decision example must include
`artifact_root`.

## Why This Won

The operator may use the runbook during the physical WING session. The visible
example must not produce a record that the ledger rejects.

## Rejected Alternatives

Implicit reliance on generated templates was rejected because it leaves the
manual instruction path stale.

## Implementation Plan

- Update `Docs/runbooks/WING_READONLY_PREFLIGHT.md`.
- Add a rails regression in `tests/test_wing_mvp_rails.py`.

## Test Plan

- Run targeted rails and review-decision tests.
- Run focused WING/no-WING tests.
- Run full standard pytest.
- Run no-WING quick, memory lint, Project Director, and `git diff --check`.

## Risks And Rollback

Risk is limited to documentation drift. Rollback is removing the field from the
runbook, but that would restore the stale manual example.
