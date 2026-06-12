# Codex Proposal: WING MVP Rows Structured Gate

## Short thesis

The WING MVP blocker rows should name the structured handoff gate before
supervised fader/gain proof, because the tooling now validates and summarizes
that contract.

## What I understood

The runbook and release summary now require `supervised_prestart_gate` to match
`expected_supervised_prestart_gate`. The fader and gain rows in
`.ai/project-layer/wing_mvp_blockers.md` still mentioned only accepted decision
JSON and the final operator-sequence step.

## Proposed solution

- Add the structured gate match requirement to both supervised proof rows.
- Extend `tests/test_wing_mvp_rails.py` to assert the rows and generated
  progress report include both structured gate field names.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_wing_mvp_rails.py`
- `.ai/briefs/wing-mvp-rows-structured-gate-20260612.md`
- `.ai/proposals/codex-wing-mvp-rows-structured-gate-20260612.md`
- `Docs/adr/wing-mvp-rows-structured-gate-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_ROWS_STRUCTURED_GATE_20260612.md`

## Alternatives considered

- Leave blocker rows unchanged because the runbook has the detail. Rejected
  because the blocker table is the project director source of truth.
- Add only `supervised_prestart_gate`. Rejected because the safety check is the
  equality with the expected gate.

## Risks

The row text is longer. That is acceptable because it is a safety gate for real
WING proof execution.

## Test plan

- Focused WING MVP rails test.
- Regenerate and inspect latest progress report.
- Affected WING/no-WING tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may prefer a compact symbolic gate name. I would keep the explicit field
names until there is a schema reference that operators and agents can rely on.
