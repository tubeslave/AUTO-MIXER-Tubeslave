# WING MVP Handoff Expected Artifacts Next Action

## Context

The WING read-only handoff packet and no-WING summary now expose
`expected_artifacts`, `expected_artifact_globs`, `plan_expected_artifacts`,
`expected_artifacts_match_plan`, `summary_glob_matches_plan`, and
`event_log_glob_matches_plan`.

The canonical first blocker still needs to require those fields before any
supervised fader/gain proof planning, otherwise future Goal-mode cycles can
preserve the older selected action and miss the artifact-list drift gate.

## Task

Promote the expected-artifacts handoff checks into the WING MVP selected action,
MVP/runbook/autoloop docs, generated progress report, docs index, and rail
tests.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure from offline packet evidence.
- Keep the first blocker as `WING read-only connection proof`.

## Test Plan

- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run focused Project Director and WING rail tests.
- Run direction guard and Project Director JSON checks.
- Run `git diff --check` and memory lint.
