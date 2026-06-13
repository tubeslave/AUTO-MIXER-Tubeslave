# ADR: WING MVP Review Report Output Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Before readiness or supervised proof can proceed, a clean read-only preflight
candidate must be manually reviewed, and the operator-facing reports must point
to the intended files.

The review decision template path is now contract-owned, but the artifact-review
and decision-review report output paths were still duplicated across handoff and
no-WING tooling.

## Options Considered

1. Keep report output paths duplicated.
2. Move the operator-facing report output paths into the shared WING preflight
   contract and validate handoff packet values.
3. Change only the review tools' private defaults.

## Decision

Use option 2. The shared preflight contract now owns the operator-facing
artifact-review and decision-review report output paths, and no-WING handoff
artifact validation fails closed when packet fields, declared flags, or command
`--output` values drift.

## Why This Won

The handoff packet is the operator-visible proof chain. It must make the report
paths machine-checkable so operators do not review stale or unexpected files
before a readiness or supervised-proof step.

## Rejected Alternatives

Duplicated literals were rejected because they allow drift. Tool-default-only
changes were rejected because they do not validate already-generated or edited
handoff packets.

## Implementation Plan

- Add shared report output constants to `backend/wing_readonly_preflight_contract.py`.
- Make handoff/no-WING command builders consume those constants.
- Add packet match flags for artifact-review and decision-review outputs.
- Validate packet fields, match flags, and review command outputs in no-WING.
- Add happy-path and mismatch tests.

## Test Plan

- Compile touched modules.
- Run focused handoff/no-WING/director/guard/rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Risks And Rollback

The main risk is stale fixture or summary consumers. Rollback would restore the
duplicated literals, but that would reopen the path where handoff review reports
can drift without the release gate catching it.
