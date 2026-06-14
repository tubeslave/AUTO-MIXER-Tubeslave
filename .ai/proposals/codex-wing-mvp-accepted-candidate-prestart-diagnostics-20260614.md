# Codex Proposal: WING MVP Accepted-Candidate Prestart Diagnostics

## Short Thesis

Keep the accepted-candidate prestart gate fail-closed, but make it explainable
by surfacing per-source contract violations wherever the operator reviews the
decision path.

## What I Understood

The active blocker remains `WING read-only connection proof`. The system must
not proceed to supervised fader/gain proof unless accepted review source ids
have a valid accepted-candidate prestart trace, including expected artifact
bindings and a passed no-WING prestart summary.

## Proposed Solution

Add a shared contract helper that returns prestart-contract violation names by
accepted source id. Use it in the review decision ledger and supervised proof
plan, then render the resulting maps in the decision-review CLI and no-WING
summary.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- `tools/no_wing_release_check.py`
- WING decision/proof/no-WING tests and runbook docs.

## Alternatives Considered

- Keep only the boolean gate: rejected because it forces manual reverse
  engineering of the missing trace field.
- Add a separate diagnostic tool: rejected because the existing ledger/report
  path is already the operator-facing gate.

## Risks

- Long field names can make reports noisier.
- Diagnostics must not be treated as authority to repair or execute anything.

## Test Plan

- Focused pytest for review decision, decision-review CLI, supervised proof
  plan, no-WING release check, Project Director, and WING MVP rails.
- no-WING quick validation.
- Direction guard and memory lint.
- Full test suite before commit.

## Where Kimi May Disagree

Kimi may prefer moving all diagnostics into a richer decision schema. I prefer a
small shared helper first because it reduces ambiguity without changing the
operator decision artifact format.
