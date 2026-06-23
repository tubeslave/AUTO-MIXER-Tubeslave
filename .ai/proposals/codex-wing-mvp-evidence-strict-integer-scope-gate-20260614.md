# Proposal: WING MVP Evidence Strict Integer Scope Gate

## 1. Short Thesis

Make WING read-only evidence intake reject numeric strings and floats in channel
scope fields instead of coercing them to integers.

## 2. What I Understood

The project is still blocked on the real WING read-only connection proof. While
hardware evidence is unavailable, useful local work is tightening the artifact
gate that will decide whether a real capture is safe to send to manual review.

## 3. Proposed Solution

Change the shared `_int_tuple()` helper in
`backend/wing_readonly_preflight_evidence.py` to accept only `int` values and
continue rejecting `bool`. Add a regression test proving string and float
channel scope values fail closed.

## 4. Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `Docs/adr/wing-mvp-evidence-strict-integer-scope-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_EVIDENCE_STRICT_INTEGER_SCOPE_GATE_20260614.md`

## 5. Alternatives Considered

Keep coercion and rely on later readback-target checks. Rejected because scope
typing should be explicit at the candidate boundary.

Add a full JSON schema dependency. Rejected because the existing local
validator can enforce this without new production dependencies.

## 6. Risks

Low. Real runner output already writes channel ids as integers. This only
rejects malformed summaries that previously relied on coercion.

## 7. Test Plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_no_wing_release_check.py tests/test_project_director_agent.py`
- `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## 8. Where The Other Agent May Disagree

Kimi may prefer schema-wide strict typing for every artifact count and id. That
is a reasonable follow-up, but this patch addresses the live-proof scope gate
with minimal blast radius.
