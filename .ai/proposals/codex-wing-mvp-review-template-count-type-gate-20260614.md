# Proposal: WING MVP Review Template Count Type Gate

## 1. Short Thesis

Stop the WING read-only review decision template from coercing malformed
readback counts into valid integers.

## 2. What I Understood

The current MVP is still blocked on real WING read-only proof. While waiting
for hardware evidence, local work should harden the offline candidate and manual
review rails so a later real capture cannot be promoted through weak typing.

## 3. Proposed Solution

Add a `_non_bool_int()` helper to `tools/review_wing_readonly_preflight_artifacts.py`
and use it for review rows, decision template creation, and markdown rendering.
Add a regression test that mutates a clean report with boolean/string count
values and verifies template generation fails closed.

## 4. Likely Files To Touch

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `Docs/adr/wing-mvp-review-template-count-type-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_COUNT_TYPE_GATE_20260614.md`

## 5. Alternatives Considered

Rely on evidence intake normalization. Rejected because the template builder
also accepts already-built report mappings and should defend its own boundary.

## 6. Risks

Low. Valid evidence rows already carry integer counts. The change only rejects
malformed or hand-edited review data earlier.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may prefer schema-wide validation for every report mapping. That would be
useful later, but this patch closes the immediate manual-decision template gap
with minimal blast radius.
