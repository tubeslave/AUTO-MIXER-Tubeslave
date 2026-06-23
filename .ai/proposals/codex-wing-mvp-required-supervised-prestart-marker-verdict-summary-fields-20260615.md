# Codex Proposal: WING MVP Required Supervised Prestart Marker Verdict Summary Fields

## Short Thesis

Add explicit no-WING verdicts for the top-level required supervised prestart
marker list, proving it matches both the shared contract and the structured
supervised gate marker list.

## What I Understood

The current WING MVP remains fail-closed until a real read-only preflight
artifact is captured and manually reviewed. Offline handoff summaries should
make every pre-supervised-proof condition directly auditable.

## Proposed Solution

- Compare `required_supervised_prestart_gate_markers` to the shared supervised
  proof gate marker contract.
- Compare it to `supervised_prestart_gate.required_markers`.
- Render both verdicts in `summary.json` and `summary.md`.
- Add tests and Project Director baseline markers.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_project_director_agent.py`
- `tools/codex_autoloop/update_mvp_progress.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`

## Alternatives Considered

- Rely on structured gate verdicts only. Rejected because top-level marker-list
  integrity remains implicit.
- Replace list checks with count checks. Rejected because count-only validation
  is weaker than marker-list equality.

## Risks

- More fields can make summaries longer. Mitigation: keep them adjacent to the
  existing marker-list and marker-count fields.

## Test Plan

- Compile touched Python files.
- Run focused WING/no-WING tests.
- Run no-WING quick validation.
- Run full pytest.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer a generalized marker verdict helper. The direct projection keeps
this live-sound gate change narrow and auditable.
