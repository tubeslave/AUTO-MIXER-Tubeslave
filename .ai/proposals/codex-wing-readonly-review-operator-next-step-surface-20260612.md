# Proposal: Surface blocked-runner next steps in WING review

## Short Thesis

Carry `operator_next_step` from preflight summaries into evidence and review
artifact rows so the operator can see why a blocked WING proof must be replaced.

## What I Understood

The first MVP blocker is still real WING read-only connection proof. Offline
tooling must fail closed and make the next required operator action visible
without contacting WING or granting write authority.

## Proposed Solution

- Add `operator_next_step` to `WingReadonlyPreflightArtifactRecord`.
- Populate it from validated summary payloads, including invalid summaries.
- Render it in `tools/review_wing_readonly_preflight_artifacts.py`.
- Add a regression for an `expected_plan_mismatch` summary.

## Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`

## Alternatives Considered

Keeping the field only in runner/no-WING output was rejected because archived
review reports would hide the repair instruction from the manual WING evidence
review path.

## Risks

The field is diagnostic only. The main risk is accidentally treating guidance
as permission; tests keep `live_authority_granted` false.

## Test Plan

- Focused review/evidence tests.
- Adjacent WING/no-WING gate tests.
- Full repository pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer adding a top-level report summary field. I kept the change at
the artifact-row level because multiple artifacts can have different blockers.
