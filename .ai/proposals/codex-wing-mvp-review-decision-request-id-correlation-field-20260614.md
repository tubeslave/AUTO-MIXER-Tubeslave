# Codex Proposal: WING MVP Review Decision Request-Id Correlation Field

## 1. Short thesis

Persist request-id correlation evidence in WING read-only preflight decision
records so accepted operator decisions cannot omit the request/response binding
that proves readbacks belong to the reviewed queries.

## 2. What I understood

The artifact validator checks request ids and the review report exposes the
canonical policy. The durable decision JSON should carry the same policy and a
true correlation-required flag because it becomes the operator acceptance
surface for later supervised proof planning.

## 3. Proposed solution

Add `request_id_policy` and `request_id_correlation_required` to the shared
decision-template required fields, emit them in pending decision templates, and
reject decision records where the policy is not canonical or the correlation
flag is not true.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`

## 5. Alternatives considered

- Rely on artifact validity only: rejected because decision records can be
  reviewed separately from the report.
- Rely on file hashes only: rejected because hashes bind bytes, not the
  semantic request/readback correlation contract.

## 6. Risks

Existing decision JSON records without the new fields will fail closed and must
be replaced. That is acceptable for real WING readiness evidence.

## 7. Test plan

Run focused review-decision, decision-tool, review-tool, handoff, no-WING, WING
rail, and Project Director tests, then standard pytest, no-WING quick runtime,
`git diff --check`, memory lint, and Project Director.

## 8. Where the other agent may disagree

Kimi may argue this duplicates artifact validation. The counterpoint is that
operator acceptance should snapshot the proof-critical request-id contract, not
only refer to a mutable external review surface.
