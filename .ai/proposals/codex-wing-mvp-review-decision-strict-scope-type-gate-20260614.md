# Proposal: WING MVP Review Decision Strict Scope Type Gate

## 1. Short Thesis

Make the WING read-only review decision ledger reject string and float channel
scope values instead of coercing them into accepted integers.

## 2. What I Understood

The current MVP is still the supervised WING Rack proof, blocked on real
read-only hardware capture and operator review. Local progress should continue
tightening the proof rail without granting live authority.

## 3. Proposed Solution

Update `_int_list()` in `backend/wing_readonly_preflight_review_decision.py` so
it accepts only real `int` values and excludes `bool`. Add a regression test
that passes `"3"` and `3.0` inside an otherwise accepted decision record and
expects ledger replacement.

## 4. Likely Files To Touch

- `backend/wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `Docs/adr/wing-mvp-review-decision-strict-scope-type-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_STRICT_SCOPE_TYPE_GATE_20260614.md`

## 5. Alternatives Considered

Rely on evidence/template gates. Rejected because manual decision records can be
loaded independently and the ledger should defend its own acceptance boundary.

## 6. Risks

Low. Generated templates already write integer scope values. The change only
rejects malformed or hand-edited decision records earlier.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may prefer a shared strict JSON-schema layer for every WING artifact and
decision record. That is reasonable later, but this patch closes the immediate
manual-decision acceptance gap without changing unrelated surfaces.
