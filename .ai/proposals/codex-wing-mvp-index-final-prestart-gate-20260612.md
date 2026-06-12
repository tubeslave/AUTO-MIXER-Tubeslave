# Codex Proposal: WING MVP Index Final Prestart Gate

## Short thesis

The main docs index should link the newest read-only runbook final prestart
gate, because operators use the index to find current WING MVP rails.

## What I understood

`Docs/runbooks/WING_READONLY_PREFLIGHT.md` now requires the final handoff
operator-sequence step to carry the accepted-candidate prestart gate before
supervised proof. `Docs/INDEX.md` still pointed to the older supervised runbook
prestart report and latest progress report, but not the new read-only runbook
gate report.

## Proposed solution

- Add `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md`
  to the reports list.
- Add the same report to the `Current WING MVP rails` row.
- Extend `tests/test_wing_mvp_rails.py` to assert the index keeps the report and
  the final-step prestart wording.

## Likely files to touch

- `Docs/INDEX.md`
- `tests/test_wing_mvp_rails.py`
- `.ai/briefs/wing-mvp-index-final-prestart-gate-20260612.md`
- `.ai/proposals/codex-wing-mvp-index-final-prestart-gate-20260612.md`
- `Docs/adr/wing-mvp-index-final-prestart-gate-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_INDEX_FINAL_PRESTART_GATE_20260612.md`

## Alternatives considered

- Leave the index unchanged. Rejected because the canonical docs entrypoint
  would hide the latest safety gate.
- Add only the report-list row. Rejected because the product/agent rails table
  is the faster path operators and agents use.

## Risks

The main risk is index churn. The regression is intentionally narrow and checks
only current WING MVP safety navigation.

## Test plan

- Focused `tests/test_wing_mvp_rails.py` index test.
- Full `tests/test_wing_mvp_rails.py`.
- Memory lint and full pytest before commit.

## Where the other agent may disagree

Kimi may argue the reports list is enough. I would keep the rails-row link too,
because the final prestart gate is an active operator boundary, not only a
historical report.
