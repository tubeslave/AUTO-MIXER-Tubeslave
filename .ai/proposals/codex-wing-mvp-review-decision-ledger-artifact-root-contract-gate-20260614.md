# Proposal: WING MVP Review Decision Ledger Artifact Root Contract Gate

## 1. Short thesis

Move accepted-candidate artifact-root contract visibility into the review
decision ledger so stale accepted decisions cannot look like manual readiness
candidates before proof-plan validation runs.

## 2. What I understood

The current MVP rail is the WING read-only proof path. Accepted operator review
decisions must not imply readiness when their candidate metadata can no longer
prove canonical artifact-root binding.

## 3. Proposed solution

Compute accepted-candidate artifact-root contract fields inside
`build_wing_readonly_preflight_review_decision_ledger()`, include them in the
ledger summary, and require them before
`manual_readiness_update_candidate_available` can become true.

## 4. Likely files to touch

- `backend/wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- Council ADR/report/index files

## 5. Alternatives considered

- Leave artifact-root checks only in proof-plan/no-WING validation.
- Make the shared helper tolerate all synthetic source ids globally.
- Require every inline test payload to use a file-like source id.

## 6. Risks

The ledger supports inline artifact payloads in tests, whose `source_id` is
synthetic. The implementation should tolerate that at ledger level while
leaving file-backed proof-plan and no-WING path checks strict.

## 7. Test plan

Run focused review-decision/proof-plan/no-WING tests, WING-focused tests,
standard pytest, no-WING release check, diff check, memory lint, and Project
Director.

## 8. Where the other agent may disagree

Kimi may prefer a shared source-id abstraction. For this blocker, a local
ledger wrapper is smaller and avoids weakening the strict shared helper used by
proof-plan and no-WING validation.
