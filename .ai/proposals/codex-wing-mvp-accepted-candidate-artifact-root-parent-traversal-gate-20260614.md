# Proposal: WING MVP Accepted Candidate Artifact Root Parent Traversal Gate

## 1. Short thesis

Normalize accepted-candidate artifact paths and reject parent traversal before
they can gate supervised WING proof planning.

## 2. What I understood

The current project direction remains the supervised WING Rack MVP. The first
open blocker is WING read-only connection proof, and the immediate risk is
trusting malformed read-only preflight candidate metadata before a physical
proof run.

## 3. Proposed solution

Strengthen the shared accepted-candidate artifact-root contract helper so every
candidate path is checked after POSIX normalization and any `..` segment is
reported as a violation.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`
- Council ADR/report/index files

## 5. Alternatives considered

- Keep prefix checks only.
- Add the traversal logic separately in the proof planner and no-WING validator.
- Require a broad candidate schema migration.

## 6. Risks

The main risk is rejecting a manually edited candidate that used non-canonical
relative paths. That is acceptable for this gate because read-only proof
artifacts should be repo-relative and canonical before they can protect a live
console workflow.

## 7. Test plan

Run focused tests for the review decision helper, supervised proof plan, and
no-WING release validator; then run the WING-focused suite, standard pytest,
release checks, diff check, memory lint, and Project Director.

## 8. Where the other agent may disagree

Kimi may prefer a dedicated path type or schema-level artifact path normalizer.
For this MVP blocker, the shared helper is the smaller and safer change because
both downstream surfaces already consume it.
