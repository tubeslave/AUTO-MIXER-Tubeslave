# Proposal: WING MVP Current Evidence Latest Reports

## Short thesis

Add the newest validated WING MVP reports to the first blocker evidence list so
future autonomous cycles inspect the same proof boundaries that currently govern
supervised proof planning.

## What I understood

The blocker next-action text already requires shared artifact identity and a
valid accepted-candidate prestart trace. The evidence cell did not yet cite the
reports that document those two current boundaries, so Project Director did not
surface them through `selected_evidence_files`.

## Proposed solution

- Add the read-only artifact contract constants report to the first blocker
  evidence list.
- Add the decision-review prestart ready gate report to the same evidence list.
- Assert both paths through blocker table parsing and Project Director output.
- Add a short report and ADR documenting that this is an evidence refresh only.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- Council brief/proposal/ADR/report docs.

## Alternatives considered

- Leave the evidence list as-is. Rejected because Project Director would keep
  omitting current proof-boundary reports from selected evidence.
- Teach Project Director to infer latest reports from `Docs/INDEX.md`. Rejected
  as too broad; the blocker table should remain the canonical source.

## Risks

- The long markdown table row could drift or fail parser expectations. Mitigate
  with focused parser and Project Director tests.
- This documentation refresh must not imply readiness closure.

## Test plan

- Run focused WING MVP rails, Project Director, and direction guard tests.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may prefer not adding more entries to an already long evidence cell. The
benefit is that Project Director continues to surface the exact current proof
reports named by the active blocker.
