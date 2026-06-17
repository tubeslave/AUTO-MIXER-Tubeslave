# Proposal: WING MVP Accepted Candidate Artifact Root Path Scope

## 1. Short Thesis

Move accepted-candidate artifact-root validation into the shared WING contract
and require every candidate artifact path to stay under the declared root.

## 2. What I Understood

The current supervised WING MVP must not allow physical supervised proof steps
unless the accepted read-only candidate points to the exact artifact set that
the operator reviewed.

## 3. Proposed Solution

- Add `wing_readonly_preflight_accepted_candidate_artifact_root_contract()` to
  `backend/wing_readonly_preflight_contract.py`.
- Require `source_id`, `summary_path`, `events_path`,
  `expected_summary_artifact`, and `expected_events_artifact` to live under
  `artifact_root`.
- Switch `backend/supervised_wing_proof_plan.py` and
  `tools/no_wing_release_check.py` to the shared helper.
- Add regressions where prestart remains valid but artifact paths leave the
  declared root.

## 4. Likely Files To Touch

- `backend/wing_readonly_preflight_contract.py`
- `backend/supervised_wing_proof_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_supervised_wing_proof_plan.py`
- `tests/test_no_wing_release_check.py`

## 5. Alternatives Considered

Leaving duplicate local helpers is cheaper but risks drift between planner and
release validation. A broader schema refactor is larger than needed for this
gate.

## 6. Risks

The stricter helper may reject incomplete hand-built fixtures. That is
acceptable for accepted-candidate proof planning because real accepted
candidates must bind the reviewed artifact paths.

## 7. Test Plan

Run focused supervised proof-plan, no-WING, and review-decision tests, then run
the broader WING-focused suite and the standard repository pytest command.

## 8. Where The Other Agent May Disagree

Kimi may argue for exact root equality rather than under-root checks. That is a
reasonable later tightening, but under-root validation closes the current stale
path escape without changing legitimate nested artifact layouts.
