# WING MVP Proof Plan Canonical Prestart Helper

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The canonical read-only preflight contract already defines the accepted
candidate prestart trace requirements, including expected artifact binding,
backend prestart command fields, prestart summary path/SHA/status, and no-live
authority booleans.

## Task

Make `tools/no_wing_release_check.py` validate the `next_wing_proof_plan`
accepted-candidate prestart trace through the shared
`backend.wing_readonly_preflight_contract` helper instead of a shorter local
manual check.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure from offline or candidate evidence.
- Keep the selected blocker as `WING read-only connection proof`.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused no-WING release-check tests.
- Run focused WING/readiness rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run `py_compile`, memory lint, `git diff --check`, and the standard full
  pytest command before commit.
