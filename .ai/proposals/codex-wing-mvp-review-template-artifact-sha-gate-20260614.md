# Proposal: WING MVP Review Template Artifact SHA Gate

## 1. Short Thesis

Stop the WING read-only review decision template from accepting malformed
artifact digest fields when it is given an already-built report mapping.

## 2. What I Understood

The current MVP remains blocked on physical WING read-only proof and operator
review. While waiting for hardware evidence, the manual-review rails should
reject malformed candidate identity before a pending decision template is
generated.

## 3. Proposed Solution

Use the shared `is_sha256_hexdigest()` contract helper in
`tools/review_wing_readonly_preflight_artifacts.py`. Require both
`summary_sha256` and `events_sha256` to be valid SHA-256 hex strings before
creating a pending review decision template.

## 4. Likely Files To Touch

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `Docs/adr/wing-mvp-review-template-artifact-sha-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_ARTIFACT_SHA_GATE_20260614.md`

## 5. Alternatives Considered

Rely on evidence intake and ledger validation. Rejected because the template
builder accepts report mappings directly and should defend its own manual
review boundary.

## 6. Risks

Low. Valid read-only preflight reports already carry real SHA-256 digests. The
patch only rejects malformed or hand-edited review data earlier.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may prefer a single full schema validator for report mappings. That is
reasonable later, but this patch closes the immediate artifact SHA template gap
with the same narrow fail-closed pattern as the count, scope, and value-type
gates.
