# Codex Proposal: WING Read-Only Review Listener Checklist Regression

## Short thesis

Make the listener confirmation checklist a required visible field in review and
decision review reports, not only in the raw no-WING busy-port artifact.

## What I understood

The no-WING prestart gate now emits `listener_confirmation_checklist` when the
backend port is busy. The next risk is that downstream review artifacts keep
passing while only showing the older `listener_processes` field.

## Proposed solution

- Add regression assertions to the read-only preflight artifact review report.
- Add regression assertions to the review-decision report.
- Add no-WING summary assertions for decision-ledger candidate review output.

## Likely files to touch

- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Add more production rendering code. Rejected because the existing renderers
  already include the full prestart failure-detail field list.
- Leave the checklist covered only by raw no-WING tests. Rejected because the
  operator will review downstream reports during the hardware step.

## Risks

Low. This is regression coverage over existing generated surfaces.

## Test plan

- Review/no-WING focused tests.
- Adjacent WING/no-WING suite.
- `git diff --check`, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may consider this redundant. I would keep it because the project’s current
blocker is an operator-reviewed hardware proof, so downstream visibility is
part of the proof boundary.
