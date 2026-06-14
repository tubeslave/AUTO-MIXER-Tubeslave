# WING MVP Handoff Proof-Plan Summary Fields

## Context

The first open WING MVP blocker remains `WING read-only connection proof`.
The supervised proof plan and no-WING summary now expose canonical
accepted-candidate prestart diagnostics, but the operator handoff packet did
not explicitly list the proof-plan summary fields that must be checked before
moving from read-only proof to supervised proof.

## Task

Bind the WING read-only handoff packet to the required supervised proof-plan
summary fields, including the canonical expected candidate prestart source IDs
and violations map. The no-WING release gate must fail closed if a handoff
packet omits or weakens that list.

## Constraints

- No backend startup.
- No WING transport usage.
- No console reads or writes.
- No readiness closure from offline or candidate evidence.
- Keep the selected blocker as `WING read-only connection proof`.
- Keep live authority denied until real WING artifacts are reviewed.

## Test Plan

- Run focused handoff/no-WING tests.
- Run focused WING/readiness rail tests.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Run Project Director and direction guard.
- Run no-WING quick check and expect fail-closed if a listener is present.
- Run `py_compile`, memory lint, `git diff --check`, and the standard full
  pytest command before commit.
