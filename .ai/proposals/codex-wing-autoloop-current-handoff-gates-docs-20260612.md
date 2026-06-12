# Proposal: WING Autoloop Current Handoff Gates Docs

## Short thesis

Keep Codex autoloop prompt and README aligned with the latest WING handoff gate
fields before future cycles select work.

## What I understood

Project Director selected action now requires the handoff blocker anchor to
state it does not close the blocker and requires the final supervised operator
step to carry accepted-decision, accepted-candidate prestart-trace, and proof
runbook markers.

## Proposed solution

- Update `tools/codex_autoloop/codex_cycle_prompt.md`.
- Update `tools/codex_autoloop/README.md`.
- Extend Project Director prompt/README tests for the new fields.
- Record the change in ADR/report/index surfaces.

## Likely files to touch

- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`
- `Docs/adr/wing-autoloop-current-handoff-gates-docs-20260612.md`
- `Docs/reports/AUTOMIXER_WING_AUTOLOOP_CURRENT_HANDOFF_GATES_DOCS_20260612.md`

## Alternatives considered

- Leave only the blocker table updated. Rejected because autonomous cycles read
  the autoloop prompt before selecting a WING task.
- Add a new runtime gate. Rejected because this is a documentation/test sync for
  already enforced handoff fields.

## Risks

Low. The change does not alter runtime behavior or live authority. The main risk
is prompt drift, covered by focused tests.

## Test plan

- Focused Project Director and WING MVP rail tests.
- Project Director and direction guard CLIs.
- `git diff --check`, memory lint, and full pytest before commit.

## Where the other agent may disagree

Kimi may view the prompt text as redundant with the blocker table. I would keep
it because the autoloop prompt is the pre-selection instruction surface.
