# Brief: WING read-only runbook handoff plan output

## Context

The WING read-only handoff packet has a default `plan_output`, and no-WING
checks validate the same path. The runbook's optional handoff command did not
show `--plan-output`, making the expected-plan artifact less explicit for the
operator.

## Task

Make the runbook handoff command explicitly point at the same expected plan
artifact used by the plan-only and capture commands.

## Acceptance

- The handoff example includes `--plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json`.
- Rails tests protect the explicit plan-output path.
- No WING/backend/OSC contact is introduced.
