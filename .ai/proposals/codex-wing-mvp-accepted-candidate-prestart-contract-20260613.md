# Proposal: WING MVP accepted-candidate prestart trace contract

## 1. Short thesis

Centralize the accepted-candidate no-WING prestart trace check so review
decisions and supervised proof planning cannot disagree.

## 2. What I understood

The current first MVP blocker remains WING read-only connection proof. Offline
work is still useful only when it tightens the real WING proof path and keeps
supervised proof blocked until real artifacts are reviewed.

## 3. Proposed solution

- Add shared prestart summary constants and a helper in
  `backend/wing_readonly_preflight_contract.py`.
- Replace local accepted-candidate prestart trace logic in
  `backend/wing_readonly_preflight_review_decision.py`.
- Replace the duplicate proof-plan gate in
  `backend/supervised_wing_proof_plan.py`.
- Bind runner/evidence prestart summary constants to the shared contract.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/wing_readonly_preflight_review_decision.py`
- `backend/supervised_wing_proof_plan.py`
- `backend/wing_readonly_preflight_evidence.py`
- `tools/wing_readonly_preflight_runner.py`
- focused tests under `tests/`

## 5. Alternatives considered

Keep both implementations and add cross-tests. That still leaves two code paths
to maintain, so it is weaker than one shared helper.

## 6. Risks

The helper could become too broad and accidentally inspect files or imply
readiness. Keep it pure and report-only: it checks only candidate review row
fields already present in summaries.

## 7. Test plan

- `py_compile` on touched modules.
- Focused pytest for WING read-only review decision, supervised proof plan,
  runner, evidence, and no-WING release check.
- no-WING quick release check with isolated artifact root.

## 8. Where the other agent may disagree

Kimi may prefer keeping the helper private to the review decision ledger. The
counterpoint is that the proof plan is exactly the second consumer that must
agree before any supervised proof can be planned.
