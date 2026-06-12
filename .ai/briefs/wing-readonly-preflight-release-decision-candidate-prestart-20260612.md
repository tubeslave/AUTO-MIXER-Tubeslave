# WING Read-Only Preflight Release Decision Candidate Prestart

## Context

The WING read-only preflight review decision ledger can now carry candidate review rows with backend-port prestart contract fields. The no-WING release summary still needs to preserve those rows so an operator can audit the prestart safety contract from the release artifact without starting a backend or contacting a WING console.

## Task

Surface `candidate_reviews` from the `wing_readonly_preflight_review_decision_ledger_schema` detail into the no-WING release summary JSON and markdown.

## Constraints

- Keep this no-WING and read-only.
- Do not grant live authority, execution permission, readiness closure, or console mutation.
- Preserve the existing operator review requirement.
- Add focused regression coverage.

## Test Plan

- Focused pytest for the no-WING release decision summary.
- Affected WING/no-WING pytest stack.
- Full backend pytest suite.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
