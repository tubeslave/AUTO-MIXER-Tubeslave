# Codex Proposal: WING MVP Supervised Prestart Step Text Summary Fields

## Short Thesis

Add a top-level no-WING summary field and verdict proving the structured
supervised prestart step text matches the operator sequence step.

## What I Understood

The WING MVP is still blocked on real read-only evidence. The offline handoff
rail should make every pre-supervised-proof condition directly reviewable in
summary surfaces.

## Proposed Solution

- Derive `supervised_prestart_gate_operator_sequence_step` from the structured
  gate.
- Compare it to `operator_sequence_supervised_prestart_gate_step`.
- Render the value and verdict in `summary.json` and `summary.md`.
- Add focused assertions and Project Director markers.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `tests/test_project_director_agent.py`

## Alternatives Considered

- Keep the text only inside the nested gate and operator sequence. Rejected
  because release summaries should expose a direct drift verdict.
- Generalize all string-match verdicts. Deferred to avoid refactor risk.

## Risks

- A hardcoded verdict could hide drift. Mitigation: derive from the structured
  gate and compare to the existing operator sequence field.

## Test Plan

- Compile touched Python files.
- Run focused WING/no-WING tests.
- Run no-WING quick validation.
- Run full pytest.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer a generalized step contract helper. The direct projection keeps
the active blocker change narrow and auditable.
