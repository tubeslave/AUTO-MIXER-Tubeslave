# WING Read-Only Validate Inputs Handoff Step

## Task

Add the existing WING read-only runner `--validate-run-inputs-only` mode to the official operator handoff path before real capture.

## Context

The runner can validate a real-run command against the no-WING prestart summary and expected plan without opening transport. The handoff packet and no-WING command binding surface still jumped from offline plan preview to capture.

## Constraints

- Keep the change read-only and operator-gated.
- Do not open WING transport or grant live authority.
- Preserve the selected MVP blocker: WING read-only connection proof.
- Touch only WING preflight contract, packet, no-WING surfaces, tests, and decision docs.

## Test Plan

- Compile edited Python files.
- Run WING handoff packet tests.
- Run no-WING focused WING preflight tests.
- Run full pytest before commit.
