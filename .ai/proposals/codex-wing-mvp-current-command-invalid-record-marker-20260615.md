# Proposal: WING MVP Current-Command Invalid-Record Marker

## Short Thesis

Project Director should not consider Current First Command complete unless it
requires the proof-plan invalid review-decision records field.

## What I Understood

The WING MVP remains blocked on real read-only WING evidence and manual review.
The latest proof-plan reporting work added
`readonly_preflight_review_decision_invalid_records`, but the director's
current-command baseline still allowed an older instruction that omitted that
field.

## Proposed Solution

- Add `readonly_preflight_review_decision_invalid_records` to
  `CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`.
- Update Current First Command text to require the field in
  `required_supervised_proof_plan_summary_fields`.
- Update Codex autoloop prompt and README to carry the same operator
  requirement.
- Add Project Director tests for selected action, prompt, and README coverage.

## Likely Files To Touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`

## Alternatives Considered

- Leave the marker implicit through the contract constant. Rejected because
  Project Director baseline coverage checks visible operator-command text.
- Add a broader schema diff detector. Rejected for this pass because the
  current gap is a concrete missing marker in the selected directive.

## Risks

Risk is limited to guidance text and baseline coverage. If the marker is absent,
Project Director now reports incomplete current-command coverage instead of
silently accepting stale guidance.

## Test Plan

Run targeted Project Director tests, no-WING quick, Project Director JSON,
diff check, memory lint, and standard pytest.

## Where The Other Agent May Disagree

Kimi may prefer deriving baseline markers directly from the contract constant.
That is a reasonable future refactor; this patch keeps the existing explicit
marker pattern and adds the missing field.
