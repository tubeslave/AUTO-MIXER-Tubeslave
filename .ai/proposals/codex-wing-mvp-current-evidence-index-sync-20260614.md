# Codex Proposal: WING MVP Current Evidence Index Sync

## Short thesis

Make `Docs/INDEX.md` and the first WING blocker evidence agree for current
2026-06-14 WING MVP reports so Project Director cannot miss indexed proof
context.

## What I understood

The active MVP is still the supervised WING Rack proof. The next useful
offline step is not to claim readiness, but to keep Project Director's selected
evidence synchronized with the current indexed WING reports it should inspect.

## Proposed solution

Add the missing indexed 2026-06-14 WING MVP report paths to
`.ai/project-layer/wing_mvp_blockers.md`. Then add tests that derive the
expected report set from `Docs/INDEX.md` and assert both blocker evidence and
Project Director `selected_evidence_files` contain all of it.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/INDEX.md`
- Council trail files for this task.

## Alternatives considered

- Keep adding one-off report assertions.
- Teach Project Director to infer evidence directly from `Docs/INDEX.md`.

## Risks

The index-sync rail can fail if a future index entry is intentionally outside
the first WING blocker evidence. The current scope is limited to
`AUTOMIXER_WING_MVP_*_20260614.md`, which matches today's active WING blocker
reports.

## Test plan

Run focused rails/Director tests, Project Director JSON, direction guard JSON,
memory lint, diff check, no-WING quick check, and full pytest.

## Where the other agent may disagree

Kimi may prefer explicit individual assertions. I prefer deriving this narrow
set from the docs index because it catches the exact drift that happened here
without changing runtime behavior.
