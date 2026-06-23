# ADR: WING MVP no-WING Review Decision Ledger Artifact Root Summary Surface

## 1. Context

The WING read-only review-decision ledger now computes accepted-candidate
artifact-root contract fields and blocks manual readiness candidates when that
contract is invalid. The no-WING summary path already exposed related
proof-plan artifact-root fields, but the review-decision ledger section itself
did not surface the ledger artifact-root fields or validate the blocker at the
snapshot boundary.

## 2. Options considered

- Keep artifact-root visibility only in the proof-plan section.
- Add JSON summary fields only.
- Add JSON, markdown, and fail-closed snapshot validation.

## 3. Decision

Add accepted-candidate artifact-root contract fields to the no-WING
review-decision snapshot detail, `summary.json`, and `summary.md`, and make the
snapshot validator fail closed on missing artifact-root blocker coverage.

## 4. Why this won

The ledger section becomes self-explaining and no-WING release validation now
catches stale ledger summaries before operators reach the later proof-plan
section.

## 5. Rejected alternatives

Proof-plan-only visibility hides the reason the ledger manual-candidate flag is
false. JSON-only visibility is less useful for operator review. Recomputing only
downstream leaves the ledger summary boundary stale.

## 6. Implementation plan

- Read artifact-root contract fields from the ledger snapshot summary.
- Propagate them into no-WING summary JSON and operator markdown.
- Fail closed when an accepted candidate has invalid artifact-root contract
  without the ledger artifact-root blocker or with manual candidate still true.
- Add regression tests for baseline and invalid accepted-candidate behavior.

## 7. Test plan

Run targeted no-WING review-decision tests, focused WING tests, WING-focused
tests, standard pytest, no-WING release check, diff check, memory lint, and
Project Director.

## 8. Risks and rollback

Risk is limited to stricter offline validation. Rollback is reverting the
no-WING summary field propagation, snapshot checks, and tests.
