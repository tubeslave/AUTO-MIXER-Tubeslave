# ADR: WING MVP Accepted Candidate Expected Artifact Path Contract

## 1. Context

The WING read-only review-template path gate rejects mismatched expected
artifact paths, but the shared accepted-candidate prestart contract also feeds
the review ledger, no-WING validation, and supervised proof-plan blockers. That
shared helper still allowed stale expected paths when match flags stayed true.

## 2. Options Considered

- Rely on the review-template gate only.
- Require a broad schema migration for all candidate review rows.
- Add exact expected-path comparison in the shared prestart contract helper.

## 3. Decision

Add expected summary/event artifact path comparison to the shared
accepted-candidate prestart contract helper.

## 4. Why This Won

The shared helper is the boundary that later validation uses before allowing
physical proof steps. The change is narrow, offline, and keeps live authority
false.

## 5. Rejected Alternatives

Template-only validation leaves hand-built or stale ledgers weaker. A broad
schema migration is larger than needed for the current blocker.

## 6. Implementation Plan

- Compare `expected_summary_artifact` to `summary_path` or `source_id`.
- Compare `expected_events_artifact` to `events_path` when present.
- Preserve existing non-empty and boolean match-flag checks.
- Add regression tests for ledger and no-WING validator failure modes.

## 7. Test Plan

Run focused tests for `test_wing_readonly_preflight_review_decision.py`,
`test_supervised_wing_proof_plan.py`, and `test_no_wing_release_check.py`.
Then run the WING-focused suite and the standard repository test command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed candidate review data earlier. Rollback
is reverting the helper comparisons and related regressions if real artifacts
show a legitimate alternate path representation.
