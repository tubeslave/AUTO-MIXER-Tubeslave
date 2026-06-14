# WING MVP Proof Plan Prestart Violations Map Summary

## Context

The current first open WING MVP blocker remains `WING read-only connection
proof`. Before any supervised proof, both the decision-review surface and the
next proof-plan surface must show that accepted-candidate prestart contract
violations are empty.

`backend/supervised_wing_proof_plan.py` already places
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`
in its summary. The no-WING release-check summary did not preserve or render
that map in `next_wing_proof_plan`, and its validator did not compare the map
against the decision-review ledger.

## Task

Require the no-WING `next_wing_proof_plan` summary and markdown to expose
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`,
and fail validation if it differs from the decision-review ledger map.

## Constraints

- Do not contact WING, query console state, or start/stop backend listeners.
- Do not grant execution, live authority, readiness closure, or full autonomy.
- Keep the first open blocker as `WING read-only connection proof`.
- Keep this as an offline artifact-contract hardening change.

## Test Plan

- Run focused no-WING release-check tests around `next_wing_proof_plan`.
- Run supervised proof-plan tests.
- Run Project Director and direction guard.
- Run memory lint, `git diff --check`, and the standard full pytest command.
