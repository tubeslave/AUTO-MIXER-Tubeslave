# Codex Proposal: WING MVP Progress Prestart Trace Gate

## Short thesis

The operator-facing MVP progress rails should not point from accepted read-only
review directly to supervised proof unless the accepted source ids also carry a
valid accepted-candidate prestart trace.

## What I understood

The current project milestone is still the first supervised WING Rack blocker:
real WING read-only connection proof. Recent proof-plan gates made candidate
prestart trace validation mandatory, but the blocker table and generated status
report did not show that prerequisite.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md` so the first blocker next
  action includes the accepted-candidate prestart trace prerequisite and the
  explicit proof-plan blocker string.
- Update `Docs/MVP_WING_SUPERVISED.md` stop rules/current action with the same
  boundary.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md` from the blocker table.
- Add regression coverage in MVP rails and Project Director tests.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`

## Alternatives considered

- Leave the status rails generic and rely on the proof-plan validator. Rejected
  because the next operator action should expose the same gate before the proof
  run is planned.
- Add a new blocker row. Rejected because this is a prerequisite inside the
  first WING read-only proof chain, not a separate proof family.

## Risks

Risk is limited to status wording and tests. The main operational risk is
operator confusion from longer next-action text, but explicit fail-closed
wording is preferable at the WING handoff boundary.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py`
- `/opt/homebrew/bin/git diff --check`
- Full test suite if feasible.

## Where the other agent may disagree

Kimi may prefer a dedicated blocker row for prestart trace validation. I would
keep it inside the first row until real WING proof is accepted, because the
status is still `tooling_ready_waits_real_wing` and no new live authority exists.
