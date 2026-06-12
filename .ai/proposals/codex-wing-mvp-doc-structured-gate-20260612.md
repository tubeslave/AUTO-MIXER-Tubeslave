# Codex Proposal: WING MVP Doc Structured Gate

## Short thesis

The canonical supervised WING MVP document must name the structured handoff gate
so high-level stop rules match the runbook, blocker table, and release summary.

## What I understood

The detailed surfaces now require `supervised_prestart_gate` to match
`expected_supervised_prestart_gate`. `Docs/MVP_WING_SUPERVISED.md` still stopped
only on missing accepted-candidate prestart trace or the proof-plan blocker.

## Proposed solution

- Add a Stop-The-Line rule for missing/mismatched structured handoff gate.
- Add a Stop-The-Line rule for any `supervised_prestart_gate` that grants live
  authority.
- Add the same condition to Current Next Action.
- Extend `tests/test_wing_mvp_rails.py`.

## Likely files to touch

- `Docs/MVP_WING_SUPERVISED.md`
- `tests/test_wing_mvp_rails.py`
- `.ai/briefs/wing-mvp-doc-structured-gate-20260612.md`
- `.ai/proposals/codex-wing-mvp-doc-structured-gate-20260612.md`
- `Docs/adr/wing-mvp-doc-structured-gate-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_DOC_STRUCTURED_GATE_20260612.md`

## Alternatives considered

- Rely on the runbook only. Rejected because the MVP doc controls stop rules and
  direction.
- Move all gate details to the blocker table. Rejected because the MVP doc is
  the higher-level policy surface.

## Risks

The MVP doc now repeats gate field names from release summaries. This is
intentional because stop rules need exact evidence names.

## Test plan

- Focused MVP rails test.
- Full WING MVP rails test file.
- Affected WING/no-WING tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may prefer a named gate constant in docs. That can be added later; the
current work makes the active field-level boundary explicit.
