# Codex Proposal: WING MVP Prestart Summary Type Blocker Row

## Short thesis

Keep the first blocker table baseline aligned with the current WING command by
adding the no-WING prestart summary type marker to the table row.

## What I understood

The current first command and autoloop rails now require
`backend_port_prestart_summary_type: no_wing_release_check`. Project Director
still appends the first blocker table row as baseline action text, so the row
must carry the same marker instead of relying only on the current command.

## Proposed solution

Add the type marker to `.ai/project-layer/wing_mvp_blockers.md` in the
`WING read-only connection proof` row near the accepted-candidate prestart
summary requirement. Extend the WING MVP rail test that parses the table row so
it fails if the marker disappears from the baseline source.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_TYPE_BLOCKER_ROW_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Rely only on `## Current First Command`. Rejected because Project Director
keeps the table row as baseline safety text.

Rewrite the blocker table row into a structured source. Rejected for this slice
because it is broader and riskier than closing the immediate drift point.

## Risks

The blocker row is large and awkward to review. The mitigation is a single
marker insertion plus a focused parser test that checks the rendered row.

## Test plan

Run focused rail tests, regenerate latest progress, run Project Director and
direction guard, run `git diff --check`, run memory lint, and run the standard
full pytest command.

## Where the other agent may disagree

Kimi may argue the long markdown row should be replaced by structured data. I
agree that is a better end state, but this slice deliberately avoids a larger
format migration while the WING proof remains blocked on real artifacts.
