# Codex Proposal: WING MVP Review Decision Scope Fields

## 1. Short thesis

Persist WING read-only preflight scope in operator decision records so accepted
decisions state which channels, value types, and physical readbacks were
reviewed.

## 2. What I understood

The review artifact is already scoped by channel/value type and requires
operator channel/session confirmation. The durable decision JSON should carry
the same scope, otherwise later supervised proof planning can see an accepted
decision without a machine-readable statement of what was accepted.

## 3. Proposed solution

Add scope fields to the shared decision-template required fields, emit them in
pending decision templates, and reject decision records where those fields do
not match the reviewed candidate artifact.

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

- Rely on checklist ids only: rejected because checklist ids do not expose the
  exact channel/value-type scope.
- Rely on artifact hashes only: rejected because hashes bind files, not a
  readable scope summary for downstream gates.

## 6. Risks

Existing decision JSON without scope fields will fail closed and must be
recreated from the pending template. That is desirable for real WING readiness.

## 7. Test plan

Run focused review-decision, decision-tool, review-tool, handoff, no-WING, WING
rail, and Project Director tests, then standard pytest, no-WING quick runtime,
`git diff --check`, memory lint, and Project Director.

## 8. Where the other agent may disagree

Kimi may argue hashes already bind the scope. The counterpoint is that
operator-facing decision records are also audit artifacts and should be
machine-readable without re-hashing and re-opening the source artifact.
