# WING MVP Proof Plan Canonical Prestart Diagnostics

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The no-WING `next_wing_proof_plan` gate now validates accepted-candidate
prestart traces through the shared read-only preflight contract helper, but the
operator-facing summary only exposes the reported candidate prestart fields.

## Task

Expose the canonical expected accepted-candidate prestart source IDs and
violations map in the supervised WING proof plan summary and no-WING summary.
The release gate must still fail closed when the reported plan fields diverge
from the helper-computed expected verdict.

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
- Run no-WING quick check and expect fail-closed if a listener is present.
- Run `py_compile`, memory lint, `git diff --check`, and the standard full
  pytest command before commit.
