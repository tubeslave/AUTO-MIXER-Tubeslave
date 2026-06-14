# WING MVP Autoloop Proof-Plan Summary Fields

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The handoff packet and no-WING gate now require
`required_supervised_proof_plan_summary_fields`, but the Codex autoloop prompt
and README still named only the older supervised prestart markers.

## Task

Keep autonomous Goal-mode cycles aligned with the handoff/no-WING contract by
requiring the autoloop prompt and README to check the supervised proof-plan
summary field list before any supervised fader/gain proof.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure.
- Keep Project Director audit/planning-only.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused autoloop/Project Director/WING rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run memory lint and `git diff --check`.
- Run the standard full pytest command before commit.
