# Codex Proposal: Codex Autoloop Structured WING Gate

## Short thesis

Codex autoloop guidance should enforce the same structured WING gate as the MVP
runbook, blocker rows, index, and capability matrix.

## What I understood

The active WING proof boundary now requires `supervised_prestart_gate` to match
`expected_supervised_prestart_gate` and preserve `grants_live_authority: false`.
`tools/codex_autoloop/codex_cycle_prompt.md` and the autoloop README still
described the preflight/proof sequence without that structured gate.

## Proposed solution

- Add structured gate checks to the WING-connected sequence in
  `codex_cycle_prompt.md`.
- Add the same condition to direction-guard task selection guidance.
- Document the gate in `tools/codex_autoloop/README.md`.
- Extend existing prompt/readme tests.

## Likely files to touch

- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- `.ai/briefs/codex-autoloop-structured-wing-gate-20260612.md`
- `.ai/proposals/codex-codex-autoloop-structured-wing-gate-20260612.md`
- `Docs/adr/codex-autoloop-structured-wing-gate-20260612.md`
- `Docs/reports/AUTOMIXER_CODEX_AUTOLOOP_STRUCTURED_WING_GATE_20260612.md`

## Alternatives considered

- Leave autoloop guidance unchanged because the selected blocker row has the
  gate. Rejected because prompt guidance is what autonomous cycles actually
  read before acting.
- Mention only `supervised_prestart_gate`. Rejected because equality with
  `expected_supervised_prestart_gate` is the safety condition.

## Risks

The prompt gets longer. This is acceptable for a live-sound safety rail.

## Test plan

- Focused prompt/readme rails tests.
- Affected WING/autoloop tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may prefer a single named "structured WING gate" phrase. Explicit fields
are safer here because they map directly to release-summary evidence.
