# WING Proof Plan Accepted Candidate Prestart Trace

## Context

The WING read-only preflight path is approaching the first real read-only proof. A supervised WING proof plan can only list physical proof steps after a valid read-only preflight candidate and an accepted manual review decision. The review decision summary now includes `candidate_reviews`, but the proof plan did not require accepted source ids to match candidate rows with the backend-port prestart contract.

## Options considered

1. Continue trusting accepted source ids only.
2. Require accepted source ids to match prestart-valid `candidate_reviews` rows in the proof-plan snapshot.
3. Re-open candidate artifacts from disk inside the proof-plan builder.

## Decision

Use option 2. The proof plan computes accepted candidate source ids with the shared backend-port prestart command and failure detail fields. If all accepted source ids are not prestart-valid, `readonly_preflight_review_decision_accepted` is false and no physical proof steps are listed.

## Why this won

It closes the release-summary-to-proof-plan trace without introducing filesystem coupling, backend startup, WING queries, or live writes. It also keeps the failure mode conservative: stale snapshots remain blocked for manual review.

## Rejected alternatives

- Accepted source ids alone were rejected because they do not prove prestart safety.
- Disk revalidation was deferred because upstream gates already validate artifacts and the proof-plan builder should remain snapshot-driven.

## Implementation plan

- Import shared WING read-only preflight contract constants in `backend/supervised_wing_proof_plan.py`.
- Add summary fields for candidate prestart validity and source ids.
- Extend `tools/no_wing_release_check.py` validation and markdown rendering.
- Add regression tests for valid and missing candidate prestart traces.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_supervised_wing_proof_plan.py tests/test_no_wing_release_check.py -k "supervised_wing_proof_plan or next_wing_proof_plan"`
- Affected WING/no-WING tests.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Snapshots missing `candidate_reviews` will no longer unlock proof steps. That is an intended fail-closed migration. Rollback is safe by removing the new proof-plan summary fields and restoring accepted-source-only behavior; no runtime console behavior is affected.
