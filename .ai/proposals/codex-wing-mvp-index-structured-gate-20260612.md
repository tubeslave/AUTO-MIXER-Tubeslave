# Codex Proposal: WING MVP Index Structured Gate

## Short thesis

`Docs/INDEX.md` should point current WING MVP readers to the structured gate
reports, because those are now part of the active supervised proof boundary.

## What I understood

The main index still linked the final prestart gate report and latest progress,
but it did not link the structured handoff packet, structured summary, runbook,
or project goal reports. That made the canonical entrypoint lag behind the
current machine-readable gate.

## Proposed solution

- Add structured WING gate reports to the reports table.
- Add the structured runbook and summary reports to the Current WING MVP rails
  row.
- Extend `tests/test_wing_mvp_rails.py` to assert the index keeps these links
  and field names.

## Likely files to touch

- `Docs/INDEX.md`
- `tests/test_wing_mvp_rails.py`
- `.ai/briefs/wing-mvp-index-structured-gate-20260612.md`
- `.ai/proposals/codex-wing-mvp-index-structured-gate-20260612.md`
- `Docs/adr/wing-mvp-index-structured-gate-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_INDEX_STRUCTURED_GATE_20260612.md`

## Alternatives considered

- Leave the index unchanged because individual reports exist. Rejected because
  the index is the main documentation entrypoint.
- Link every recent WING structured report in the rails row. Rejected to keep
  the row focused on active operator/reviewer surfaces.

## Risks

The index row becomes longer. This is acceptable because it represents the
active WING safety rail.

## Test plan

- Focused docs index rails test.
- Full WING MVP rails test file.
- Affected WING/no-WING tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may prefer a separate WING MVP index page. That can follow; this patch
keeps the existing canonical index accurate today.
