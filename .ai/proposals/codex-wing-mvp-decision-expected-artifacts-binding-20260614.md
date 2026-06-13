# Proposal: Bind WING decisions to expected artifacts

## Short Thesis

An accepted WING read-only decision should not be considered prestart-ready
unless its candidate review row also proves the reviewed files match the
offline plan's expected artifacts.

## What I Understood

The supervised WING MVP is still blocked on real read-only connection proof.
The current no-WING rails can prepare the handoff and decision flow, but they
must fail closed if an accepted source id lacks the expected artifact identity
that the operator approved before the real run.

## Proposed Solution

- Extend the shared accepted-candidate prestart contract helper to require
  `expected_artifacts`, expanded expected summary/event artifact paths, and true
  summary/event match flags.
- Surface those fields in decision candidate review rows and Markdown reports.
- Add regressions proving a candidate with a valid prestart trace but missing
  expected-artifact binding remains blocked.

## Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_decisions.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`

## Alternatives Considered

A report-only warning was rejected because the decision status gates the next
supervised proof runbook. The binding must be part of the contract, not only a
human-visible hint.

## Risks

Older candidate evidence without these fields will no longer qualify as
accepted-candidate prestart-ready. That is acceptable because real WING proof
should be captured with the current runner-shaped expected plan.

## Test Plan

- Focused decision ledger and decision report tests.
- Adjacent WING/no-WING rail tests.
- Project Director and direction guard checks.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may argue that manual review can catch path mismatches. I recommend
fail-closed code enforcement because artifact identity is cheap to validate and
is a prerequisite for safe supervised proof progression.
