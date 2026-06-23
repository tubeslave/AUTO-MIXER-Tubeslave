# Codex Proposal: WING MVP Prestart Required-False Plan Match Trace

## Short thesis

Accepted-candidate prestart traces should prove both the actual denial values
and their expected-plan match verdicts before supervised proof planning can use
them.

## What I understood

The current first blocker remains WING read-only connection proof. Required
false backend-port fields are carried through evidence, review, and decision
records, but a candidate row can still omit the expected `plan_<field>` and
`<field>_matches_plan` proof for those fields.

## Proposed solution

Add shared required-false plan/match field constants, have the no-WING
backend-port check write them, have evidence intake validate recorded fields
against the referenced no-WING prestart summary, and have the accepted-candidate
contract helper reject rows where any plan field is not false or any match flag
is not true.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_evidence.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- Project Director rails, WING runbook, and focused tests.

## Alternatives considered

- Keep checking only raw false values. Rejected because stale prestart summaries
  could omit the plan-match proof while still carrying false values.
- Require packet-only `declared_*` fields in prestart rows. Rejected because
  those declarations belong to the handoff packet, not the passed no-WING
  prestart summary.

## Risks

Older candidate rows without the new fields will now fail closed. That is the
desired behavior before supervised proof planning.

## Test plan

- Focused evidence/review/decision/proof-plan/no-WING tests.
- Focused Project Director and WING rails tests.
- Compile, memory lint, `git diff --check`, and full suite.

## Where the other agent may disagree

Kimi may prefer a nested schema for required-false plan verdicts. I would keep
flat fields here because existing operator reports and Project Director rails
already use flat marker strings.
