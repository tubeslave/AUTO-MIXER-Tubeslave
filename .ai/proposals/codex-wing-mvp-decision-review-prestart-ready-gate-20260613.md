# Proposal: WING MVP Decision Review Prestart Ready Gate

## Short thesis

The decision-review CLI should return `accepted_decision_ready` only when the
accepted read-only preflight source ids are backed by a valid accepted-candidate
backend prestart trace.

## What I understood

The current proof plan blocks physical supervised proof without a candidate
prestart trace. The weaker surface is the standalone decision-review report: it
can communicate that an accepted decision is ready while the separate prestart
trace condition is still missing.

## Proposed solution

- Add the accepted-candidate prestart blocker id to the shared WING preflight
  contract.
- Have the decision ledger expose
  `accepted_candidate_prestart_contract_valid` and
  `accepted_candidate_prestart_source_ids`.
- Keep `manual_readiness_update_candidate_available` false when accepted source
  ids do not have the valid prestart trace.
- Have the decision-review CLI return a non-zero blocked status in that case.
- Surface the new fields in no-WING release summaries.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- `tools/no_wing_release_check.py`
- Focused tests and WING MVP runbook/docs.

## Alternatives considered

- Leave the decision-review CLI unchanged and rely only on the proof-plan gate.
  Rejected because operators may read the decision report before the proof plan.
- Add the gate only in the CLI. Rejected because the ledger is the shared source
  for no-WING release summaries and proof planning.

## Risks

- Older accepted decision artifacts without prestart trace fields become
  blocked until replaced by a real read-only preflight artifact with the current
  backend prestart contract. That is the intended fail-closed behavior.

## Test plan

- Focused tests for the review ledger, decision-review CLI, no-WING release
  summary, and supervised proof plan.
- Project Director and direction guard checks.
- Full backend pytest.

## Where the other agent may disagree

Kimi may argue that decision acceptance and prestart trace validity are separate
states. They remain separately visible, but the ready exit code must represent
the full gate needed before the next manual WING MVP step.
