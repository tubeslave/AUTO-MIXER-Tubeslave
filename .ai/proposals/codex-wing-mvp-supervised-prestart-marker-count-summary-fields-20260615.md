# Codex Proposal: WING MVP Supervised Prestart Marker Count Summary Fields

## Short Thesis

Add explicit no-WING count fields for the supervised prestart required markers
and prove the count matches the shared contract and top-level summary list.

## What I Understood

The current WING proof rail still requires a real read-only preflight artifact.
Offline summaries should make every supervised-proof precondition directly
auditable while preserving fail-closed live authority.

## Proposed Solution

- Derive the structured gate marker count from
  `supervised_prestart_gate.required_markers`.
- Derive the expected count from the existing marker contract.
- Compare the structured count to both the contract count and the top-level
  `required_supervised_prestart_gate_markers` count.
- Render the count and verdicts in JSON and Markdown summaries.
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

- Keep only marker-list verdicts. Rejected because reviewers still need a
  direct cardinality check.
- Add a broad marker contract registry. Deferred to avoid widening this safety
  rail change.

## Risks

- Count-only checks are weaker than list checks. Mitigation: keep the existing
  list match verdicts and add count fields as an additional summary surface.

## Test Plan

- Compile touched Python files.
- Run focused WING/no-WING tests.
- Run no-WING quick validation.
- Run full pytest.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer a generalized gate-field projection helper. The direct
projection is easier to audit for this blocker and does not disturb unrelated
release gates.
