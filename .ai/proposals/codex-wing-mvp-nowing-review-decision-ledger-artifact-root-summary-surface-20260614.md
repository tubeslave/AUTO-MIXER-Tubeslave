# Proposal: WING MVP no-WING Review Decision Ledger Artifact Root Summary Surface

## 1. Short thesis

Keep no-WING operator summaries aligned with the stricter review-decision
ledger by surfacing artifact-root contract fields before proof-plan validation.

## 2. What I understood

The active MVP remains supervised WING Rack proof. Accepted read-only review
decisions must not appear usable for manual readiness updates when artifact
roots are stale or escaping their declared root.

## 3. Proposed solution

Extend the no-WING review-decision ledger snapshot detail, summary JSON, and
operator markdown section with accepted-candidate artifact-root contract fields.
Add a fail-closed release check when an accepted candidate has an invalid
artifact-root contract but the ledger does not carry the artifact-root blocker
or still marks a manual readiness candidate as available.

## 4. Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Council ADR/report/index files

## 5. Alternatives considered

- Rely on the later `next_wing_proof_plan` markdown section only.
- Keep JSON detail but skip operator-readable markdown.
- Recompute artifact-root validity only inside proof-plan validation.

## 6. Risks

The change adds stricter no-WING validation around an offline ledger field. Risk
is limited to release checks failing earlier when the ledger summary is stale.

## 7. Test plan

Run targeted no-WING review-decision tests, focused WING tests, WING-focused
tests, standard pytest, no-WING release check, diff check, memory lint, and
Project Director.

## 8. Where the other agent may disagree

Kimi may argue the proof-plan markdown already exposes these fields. The
counterpoint is operator traceability: the ledger section should explain its
own blocked manual-candidate status before downstream proof planning.
