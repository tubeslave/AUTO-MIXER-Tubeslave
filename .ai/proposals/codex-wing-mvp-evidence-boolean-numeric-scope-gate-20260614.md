# Proposal: WING MVP Evidence Boolean Numeric Scope Gate

## 1. Short Thesis

Make WING read-only evidence intake fail closed when boolean values appear in
numeric scope fields.

## 2. What I Understood

The current MVP is still the supervised WING Rack proof. The next useful local
work is tightening the read-only preflight rail while real hardware evidence is
not available. A captured artifact must not become an accepted candidate through
Python's `bool`-as-`int` behavior.

## 3. Proposed Solution

Change the shared numeric helpers in
`backend/wing_readonly_preflight_evidence.py` so booleans are not accepted as
channel ids or counts. Add a regression test that corrupts `channels`,
`operator_confirmed_channels`, `readback_count`, `physical_readback_count`, and
a readback `channel` with boolean values and verifies the candidate is rejected.

## 4. Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `Docs/adr/wing-mvp-evidence-boolean-numeric-scope-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_EVIDENCE_BOOLEAN_NUMERIC_SCOPE_GATE_20260614.md`

## 5. Alternatives Considered

Only validate this in review-decision records. Rejected because the candidate
itself should be invalid before manual review.

Add a generic JSON schema dependency. Rejected because the repo already has
local validators and this small gate does not justify a new dependency.

## 6. Risks

Very low. Real WING channel ids and counts are integers, not booleans. The
change only rejects malformed artifacts earlier.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may ask for a broader schema-level type gate across every WING artifact
field. That is reasonable later, but this patch targets the immediate live-proof
risk without changing unrelated contracts.
