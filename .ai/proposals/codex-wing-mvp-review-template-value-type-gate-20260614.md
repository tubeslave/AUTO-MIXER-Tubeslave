# Proposal: WING MVP Review Template Value Type Gate

## 1. Short Thesis

Stop the WING read-only review decision template from accepting malformed or
unsupported value-type lists when it is given an already-built report mapping.

## 2. What I Understood

The current MVP remains blocked on physical WING read-only proof and operator
review. While waiting for hardware evidence, the manual-review rails should
reject malformed candidate identity before a pending decision template is
generated.

## 3. Proposed Solution

Use the shared WING read-only supported value-type contract in
`tools/review_wing_readonly_preflight_artifacts.py`. Normalize `value_types`
only when every entry is a non-empty supported string and no duplicate appears;
otherwise return an empty candidate value list so existing identity validation
fails closed.

## 4. Likely Files To Touch

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `Docs/adr/wing-mvp-review-template-value-type-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_VALUE_TYPE_GATE_20260614.md`

## 5. Alternatives Considered

Rely on evidence intake and ledger comparison. Rejected because the template
builder accepts report mappings directly and should defend its own manual
review boundary.

## 6. Risks

Low. Valid read-only preflight reports already use supported string value
types. The patch only rejects malformed or hand-edited review data earlier.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may prefer a full report-schema validator that validates every field at
once. That is useful later, but this patch closes the immediate value-type
template gap with the same narrow pattern used by the count and channel-scope
gates.
