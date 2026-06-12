# Proposal: Promote WING review operator next actions

## Short Thesis

When WING read-only review is blocked by an invalid artifact, the report-level
next action should use the artifact's explicit `operator_next_step`.

## What I Understood

The current MVP still waits on real WING read-only proof. Offline review tools
must make fail-closed repair steps obvious without treating offline evidence as
live readiness.

## Proposed Solution

- Collect unique artifact-level `operator_next_step` values in the review
  report.
- Use the first collected step as `next_required_action` when the review is
  blocked and no clean candidate is available.
- Render the list in Markdown for operator review.
- Keep candidate acceptance and live-authority flags unchanged.

## Likely Files To Touch

- `tools/review_wing_readonly_preflight_artifacts.py`
- `tests/test_wing_readonly_preflight_review_tool.py`

## Alternatives Considered

Only rendering the field inside artifact details was rejected because operators
first scan the summary and next-action section.

## Risks

The promoted next step must remain guidance, not permission. Existing safety
flags and the regression keep `live_authority_granted` false.

## Test Plan

- Focused review/evidence tests.
- Adjacent WING/no-WING tests.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may prefer keeping all next steps in artifact details only. I prefer the
top-level promotion because it reduces operator error during blocked preflight
repair.
