# Proposal: Bind WING real artifacts to expected-artifacts

## Short Thesis

The read-only candidate review should reject artifacts whose actual summary and
event-log paths do not match the offline plan's `expected_artifacts`.

## What I Understood

The supervised WING MVP still waits on real read-only proof. The runner already
requires an operator-approved expected plan, so the evidence layer should also
prove that the reviewed files are the exact artifact locations promised by that
plan.

## Proposed Solution

- Validate `artifact_root` and `expected_artifacts` inside the expected-plan
  payload.
- Expand `<run_id>` from the real summary and compare the resulting summary and
  event-log paths with the artifact's recorded `summary_path` and `events_path`.
- Add the expected artifact list, expanded paths, and match booleans to evidence
  records and review Markdown.
- Add regression tests for stale expected artifact globs.

## Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_review_tool.py`

## Alternatives Considered

Only checking `expected_artifacts` in the runner was rejected because stale or
hand-edited artifacts can enter the review path after capture. Adding the check
to the evidence layer keeps the manual review gate fail-closed.

## Risks

Path matching may reject older fixture-style artifacts that were not laid out
under `<artifact_root>/<run_id>/`. That is acceptable for real proof intake;
tests should use runner-shaped fixtures.

## Test Plan

- Focused evidence and review tests.
- Adjacent WING runner/no-WING rail tests.
- Direction guard and Project Director rails.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer a softer warning for path mismatch. I recommend fail-closed
rejection because this rail decides whether a real WING candidate is clean
enough for manual readiness review.
