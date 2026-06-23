# Brief: WING MVP readiness unbound preflight artifact replacement test

## Task

Align the autonomy readiness regression with the current fail-closed WING
preflight evidence contract.

## Scope

- Do not change runtime readiness code.
- Update the test that previously treated an unbound WING preflight artifact as
  a candidate.
- Require `readonly_preflight_artifact_replacement_required` when the artifact is
  missing expected-artifact binding.

## Test Plan

- Run the targeted readiness regression.
- Re-run the standard pytest command until the next real failure or completion.
