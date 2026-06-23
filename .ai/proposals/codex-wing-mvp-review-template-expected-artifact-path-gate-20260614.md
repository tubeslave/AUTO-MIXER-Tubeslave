# Proposal: WING MVP Review Template Expected Artifact Path Gate

## 1. Short Thesis

Stop the WING read-only review decision template from accepting report rows
whose summary or event artifact paths are not bound to the expected artifact
paths.

## 2. What I Understood

The current MVP remains blocked on physical WING read-only proof and operator
review. While waiting for hardware evidence, the manual-review rails should
reject stale or hand-built report mappings before a pending decision template is
generated.

## 3. Proposed Solution

Require `expected_summary_artifact == summary_path`,
`expected_events_artifact == events_path`, and both expected-artifact match
flags to be true before `build_review_decision_template()` creates a pending
decision template.

## 4. Likely Files To Touch

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `Docs/adr/wing-mvp-review-template-expected-artifact-path-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_EXPECTED_ARTIFACT_PATH_GATE_20260614.md`

## 5. Alternatives Considered

Rely on evidence intake. Rejected because the template builder accepts report
mappings directly and should defend its own manual-review boundary.

## 6. Risks

Low. Valid read-only preflight reports already carry matching expected artifact
paths. The patch only rejects malformed or hand-edited review data earlier.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may prefer a single full report-schema validator. That is useful later,
but this patch closes the immediate expected-artifact path template gap with
the same narrow fail-closed pattern as the other template gates.
