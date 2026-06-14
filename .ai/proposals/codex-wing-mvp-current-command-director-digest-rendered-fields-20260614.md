# Codex Proposal: WING MVP Current Command Director Digest Rendered Fields

## Short thesis

Make Project Director baseline coverage enforce the same rendered digest fields
that the no-WING handoff summary exposes.

## What I understood

The active MVP blocker remains `WING read-only connection proof`. The no-WING
handoff summary already renders Project Director selected-action digest fields
such as `expected_project_director_selected_action_anchor`,
`actual_project_director_selected_action_sha256`, and
`project_director_selected_action_sha256_available_matches_digest`. The current
command mentioned them, but the baseline marker list and doc assertions only
covered a shorter subset.

## Proposed solution

Add the missing rendered digest fields to
`CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`, keep `Current First Command`
explicit about the coverage marker-count field, update autoloop prompt/README,
and extend rails tests so the progress tool fails baseline coverage when the
expected selected-action digest is absent.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- Council brief, ADR, report, and index files

## Alternatives considered

Leaving the shorter marker set would keep tests green but would not protect the
rendered expected/actual digest fields from future drift. Reworking the generic
substring marker matcher was deferred because this task can close the immediate
field coverage gap with a smaller, lower-risk diff.

## Risks

The marker list is long and can become noisy. The risk is acceptable because
these fields are the operator-visible proof surface for the WING handoff rail,
and the change is limited to offline docs/tests/progress gating.

## Test plan

Run focused Project Director, WING rail, autoloop prompt, and autoloop contract
tests. Then run Project Director, direction guard, no-WING quick fail-closed,
memory lint, `git diff --check`, and the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer replacing substring marker checks with a structured parser. That
is a reasonable future cleanup, but this step keeps the diff small and directly
guards the missing rendered fields now.
