# Proposal: WING MVP Review Template Strict Scope Type Gate

## 1. Short Thesis

Stop the WING read-only review decision template from accepting malformed
channel scope lists when it is given an already-built report mapping.

## 2. What I Understood

The current MVP remains blocked on physical WING read-only proof and operator
review. While waiting for that artifact, local work should keep tightening the
offline rails so a malformed report cannot look eligible for manual decision
template generation.

## 3. Proposed Solution

Add a non-coercing integer-list helper to
`tools/review_wing_readonly_preflight_artifacts.py`. Use it for review rows and
decision template candidate identity checks so `true`, `"3"`, and `3.0` do not
survive as channel scope values.

## 4. Likely Files To Touch

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `Docs/adr/wing-mvp-review-template-strict-scope-type-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_STRICT_SCOPE_TYPE_GATE_20260614.md`

## 5. Alternatives Considered

Rely on evidence intake and the review-decision ledger. Rejected because the
template builder is a separate manual-review boundary and accepts report
mappings directly.

## 6. Risks

Low. Valid WING read-only evidence already uses integer channel scope values.
The patch only rejects malformed or hand-edited review data earlier.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may prefer a full schema validator for the whole report object. That is a
reasonable later cleanup, but this patch closes the immediate channel-scope
template gap with a smaller diff.
