# WING Proof Plan Candidate Prestart Blocker Gate

## Context

The supervised WING proof plan now binds accepted review decisions to `candidate_reviews` rows that carry the backend-port prestart contract. The no-WING release validator also needs to require an explicit blocker when accepted source ids exist but that candidate prestart trace is invalid.

## Task

Make the release gate fail if `next_wing_proof_plan` is blocked by an invalid accepted candidate prestart trace but omits the specific operator blocker.

## Constraints

- No backend startup.
- No WING transport usage.
- No OSC/WebSocket writes.
- Keep physical proof steps blocked until the accepted candidate prestart trace is valid.

## Test Plan

- Focused no-WING next-plan tests.
- Affected WING/no-WING pytest stack.
- Full backend pytest suite.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
