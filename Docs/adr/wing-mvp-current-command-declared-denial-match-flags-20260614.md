# ADR: WING MVP Current Command Declared Denial Match Flags

Date: 2026-06-14
Status: Accepted

## Context

The WING read-only handoff packet declares backend-port prestart denial
`*_matches_plan` fields, and no-WING release validation recomputes those fields
to reject stale packets. Project Director current-command baseline coverage
must also require the declared fields so the selected action does not hide
whether the packet self-checks are present.

## Options considered

1. Keep only computed denial match markers in Current First Command.
2. Require declared denial match markers only in no-WING release artifacts.
3. Require declared denial match markers in Current First Command baseline
   coverage and autoloop operator rails.

## Decision

Use option 3. Current-command baseline coverage now requires five
`declared_<field>_matches_plan: true` markers alongside the computed denial
match verdicts.

## Why this won

It keeps Project Director, autoloop instructions, and no-WING release artifacts
aligned on the same operator-visible safety contract without expanding live
scope.

## Rejected alternatives

- Computed-only markers: insufficient visibility into stale packet declarations.
- Artifact-only declared checks: lets Project Director select command text that
  omits the declared self-check contract.

## Implementation plan

- Add declared denial markers to `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Add the same markers to autoloop prompt, README, and Current First Command.
- Extend Project Director and WING rails tests.
- Record the decision and report in docs and blocker evidence.

## Test plan

- Focused Project Director and WING rails tests.
- Focused WING handoff/no-WING/Project Director/rails tests.
- Compile, memory lint, full suite, and whitespace diff.

## Risks and rollback

Older Current First Command text fails baseline coverage until updated. Roll
back by removing the declared markers from baseline coverage if a migration
path is needed, but do not treat stale command text as readiness proof.
