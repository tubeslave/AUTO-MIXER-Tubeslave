# Codex Proposal: WING MVP Supervised Rows Final Prestart Gate

## Short thesis

The supervised fader/gain blocker rows should name the final handoff prestart
gate, otherwise future selected actions can drift back to accepted-decision-only
proof planning.

## What I understood

The current first blocker remains WING read-only connection proof. The
downstream supervised proof rows are still blocked, but their next-action text
is the source for future Project Director work once the earlier rows move
forward.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md` fader and gain rows to require
  clean review, accepted decision JSON, and final handoff operator-sequence step
  with valid accepted-candidate prestart trace.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Add MVP rails assertions.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives considered

- Leave downstream rows generic until they become first blocker. Rejected
  because the table is the canonical future work queue.

## Risks

Low. This is status wording and regression coverage only.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- Affected Project Director tests.
- Full pytest suite.
- `git diff --check`.

## Where the other agent may disagree

Kimi may prefer moving this into a new blocker. I would keep it in the fader/gain
rows because it is a prerequisite for those proof actions, not a separate proof.
