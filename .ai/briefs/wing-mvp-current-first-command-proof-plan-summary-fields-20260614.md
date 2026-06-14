# WING MVP Current First Command Proof-Plan Summary Fields

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The blocker row, handoff packet, no-WING gate, and autoloop prompt now require
`required_supervised_proof_plan_summary_fields`, but the explicit
`## Current First Command` section still described the older prestart gate
checks without those required proof-plan summary fields.

## Task

Make `## Current First Command` self-contained for the handoff proof-plan
summary field contract, so Project Director selected actions do not rely on
the baseline blocker row to recover this requirement.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure.
- Keep Project Director audit/planning-only.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused Project Director/WING rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run no-WING quick fail-closed check.
- Run memory lint, `git diff --check`, and the standard full pytest command.
