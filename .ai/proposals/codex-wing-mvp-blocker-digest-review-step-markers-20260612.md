# Proposal: WING MVP Blocker Digest Review Step Markers

## Short thesis

Make Project Director require the digest-review operator step to include the
same marker strings that the runbook and no-WING validator already require.

## What I understood

The anchor digest review step is the operator-visible check that the handoff
packet still points at the current first WING blocker. The selected action
should require both the boolean and the marker strings.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md`.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Update autoloop prompt/README with the same marker requirement.
- Add focused assertions in Project Director and WING MVP rail tests.
- Record the change in ADR/report/index surfaces.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-blocker-digest-review-step-markers-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_DIGEST_REVIEW_STEP_MARKERS_20260612.md`

## Alternatives considered

- Leave the marker requirement only in the runbook. Rejected because selected
  action is the autonomous entrypoint.
- Add a new runtime validator. Rejected because the existing handoff validator
  already enforces the marker set.

## Risks

Low. This is text/test synchronization. The main risk is stale generated
progress, covered by regenerating it.

## Test plan

- Focused Project Director, WING MVP rails, and direction guard tests.
- Project Director and direction guard CLIs.
- `git diff --check`, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may consider this redundant with validator coverage. I would keep it
because future autonomous cycles read the blocker table and autoloop prompt
before reaching validator output.
