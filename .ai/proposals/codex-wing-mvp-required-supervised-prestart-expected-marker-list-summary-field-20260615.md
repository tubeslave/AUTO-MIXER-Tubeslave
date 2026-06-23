# Codex Proposal: WING MVP Required Supervised Prestart Expected Marker List Summary Field

## Short Thesis

Expose `expected_required_supervised_prestart_gate_markers` so operators can
review the expected contract list beside the actual top-level marker list.

## What I Understood

The WING MVP remains fail-closed until real read-only evidence exists. Offline
handoff summaries should make the pre-supervised-proof marker contract directly
auditable without requiring nested object parsing.

## Proposed Solution

- Derive `expected_required_supervised_prestart_gate_markers` from the shared
  supervised proof gate marker contract.
- Render it in `summary.json` and `summary.md`.
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

- Rely on count and verdict fields only. Rejected because operators still need
  the expected list itself for review.
- Duplicate the expected list under multiple aliases. Deferred to keep this
  surface minimal.

## Risks

- The summary becomes longer. Mitigation: keep the expected list adjacent to the
  actual list and existing verdicts.

## Test Plan

- Compile touched Python files.
- Run focused WING/no-WING tests.
- Run no-WING quick validation.
- Run full pytest.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer exposing a single structured marker contract object instead.
The flat field is easier for current Project Director and Markdown checks.
