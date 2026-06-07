# Brief: WING Read-Only Evidence Requires Expected-Plan Requirement Flag

## Context

The WING read-only preflight runner now requires `--expected-plan` for real
runs and writes `expected_plan_required_for_real_run: true` into the summary.
Evidence intake should enforce the same proof boundary.

## Problem

The evidence validator required the expected-plan path, SHA256, match flag, and
empty violations, but it did not require the structured
`expected_plan_required_for_real_run` field. Older or hand-written summaries
could omit that field and still look like valid candidates.

## Scope

- Reject WING read-only preflight summaries without
  `expected_plan_required_for_real_run: true`.
- Update valid artifact fixtures and focused tests.
- Keep the change offline and read-only.

## Out Of Scope

- Running WING hardware.
- Closing MVP blockers from offline evidence.
- Changing live write behavior.
