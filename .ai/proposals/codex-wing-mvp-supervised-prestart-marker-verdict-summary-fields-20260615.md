# Codex Proposal: WING MVP Supervised Prestart Marker Verdict Summary Fields

## Short Thesis

Add explicit no-WING summary verdicts that the supervised prestart required
markers match both the contract and the top-level marker list.

## What I Understood

The current WING proof rail still requires real read-only evidence. Offline
handoff summaries should make every pre-supervised-proof gate directly
reviewable without nested-object parsing.

## Proposed Solution

- Derive `supervised_prestart_gate_required_markers` from the structured gate.
- Compare it to the shared handoff marker contract.
- Compare it to `required_supervised_prestart_gate_markers`.
- Render the list and verdicts in `summary.json` and `summary.md`.
- Add tests and Project Director markers.

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

- Keep only the list. Rejected because reviewers need a direct mismatch verdict.
- Refactor all marker contracts into a registry. Deferred to keep this diff
  narrowly scoped.

## Risks

- The verdict could drift if hardcoded. Mitigation: derive it from the
  structured gate and existing top-level marker list.

## Test Plan

- Compile touched Python files.
- Run focused WING/no-WING tests.
- Run no-WING quick validation.
- Run full pytest.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer a generalized marker-verdict helper. The direct projection is
clearer for the current blocker and avoids broader refactor risk.
