# Brief: WING MVP decision-ready status contract

Date: 2026-06-13

## Task

Bind the WING read-only preflight decision-review ready status to the shared
WING preflight contract.

## Context

The current MVP remains the supervised WING Rack proof. The first open blocker
is still `WING read-only connection proof`; no offline artifact may close that
blocker or grant live authority.

The handoff/proof contract already says the final supervised prestart gate
requires `accepted_decision_ready`, but
`tools/review_wing_readonly_preflight_decisions.py` still owns that status as a
local string. If those strings drift, the review CLI and supervised gate can
disagree about whether an operator-accepted read-only candidate is ready for the
next manual step.

## Scope

- Reuse the shared contract constant for the accepted decision-ready status.
- Keep the review CLI offline and read-only.
- Add regression coverage that the review CLI status matches the supervised
  prestart gate requirement.
- Do not change readiness state, connect to WING, or widen supervised proof
  authority.

## Validation

- Focused decision-review/proof-plan tests.
- WING MVP rails tests.
- Standard test command before commit.
