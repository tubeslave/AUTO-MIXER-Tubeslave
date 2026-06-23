# Codex Proposal: WING MVP Required Supervised Prestart Marker Count Summary Fields

## Short Thesis

Make the top-level required supervised prestart marker count explicit and prove
it matches both the shared contract and the structured supervised gate count.

## What I Understood

The current WING rail remains fail-closed until real read-only hardware evidence
is captured and manually reviewed. Offline no-WING summaries should make every
pre-supervised-proof gate auditable without nested-object parsing.

## Proposed Solution

- Derive the top-level required marker count from
  `required_supervised_prestart_gate_markers`.
- Derive the expected count from the existing supervised proof gate marker
  contract.
- Compare the top-level count to the contract and structured supervised gate
  marker count.
- Render the fields in `summary.json` and `summary.md`.
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

- Keep the top-level summary count implicit. Rejected because one side of the
  comparison remains hidden.
- Replace list checks with count checks. Rejected because count-only validation
  is weaker than marker-list equality.

## Risks

- Additional fields can add review noise. Mitigation: keep them next to the
  existing supervised prestart marker fields and derive them from existing
  lists.

## Test Plan

- Compile touched Python files.
- Run focused WING/no-WING tests.
- Run no-WING quick validation.
- Run full pytest.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer consolidating marker count projections into a helper. That is a
reasonable future cleanup, but direct projection is smaller and safer for the
current live-sound gate.
