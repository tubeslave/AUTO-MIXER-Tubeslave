# Proposal: WING MVP Accepted Candidate Artifact Root Canonical Root Gate

## 1. Short thesis

Reject accepted read-only WING preflight candidates whose declared
`artifact_root` contains parent traversal before they can unlock supervised
proof planning.

## 2. What I understood

The project is still moving toward the full AI Mixing Engineer, but the current
safe MVP step is the WING read-only proof rail. Accepted candidate metadata must
be unambiguous before it can protect any later supervised fader/gain proof.

## 3. Proposed solution

Extend the shared accepted-candidate artifact-root helper with an
`artifact_root_contains_parent_traversal` violation. Reuse existing planner and
no-WING validator paths so all consumers inherit the same fail-closed result.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- Council ADR/report/index files

## 5. Alternatives considered

- Rely only on normalized child path checks.
- Add special-case planner validation.
- Require artifact roots to equal one hard-coded literal.

## 6. Risks

This may reject manually edited candidate metadata that uses non-canonical
relative roots. That is acceptable because accepted proof-gating metadata should
be canonical and reviewable before a physical console workflow.

## 7. Test plan

Run focused helper/planner/no-WING tests, WING-focused tests, standard pytest,
no-WING release check, diff check, memory lint, and Project Director.

## 8. Where the other agent may disagree

Kimi may argue for a broader path normalizer object. For this blocker, the
shared helper is the pragmatic place because the downstream surfaces already
consume it.
