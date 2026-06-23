# Codex Proposal: WING MVP Supervised Prestart Expected Artifact Trace Summary Field

## Short Thesis

Surface the existing structured supervised prestart gate as a top-level
no-WING summary field so operator review can verify the expected-artifact trace
requirement without parsing nested JSON.

## What I Understood

The WING MVP remains blocked on real read-only preflight evidence. The current
offline rail has added the expected-artifact trace requirement inside
`supervised_prestart_gate`, but the aggregate summary does not yet expose a
flat field for this specific gate.

## Proposed Solution

- Derive
  `supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace`
  from `supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace`.
- Publish it in the handoff artifact summary, top-level `summary.json`, and
  operator-readable `summary.md`.
- Add focused assertions and Current First Command baseline marker coverage.
- Add a report and ADR entry documenting that this remains no-authority
  offline validation.

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

- Leave the marker nested only. Rejected because operator and automation checks
  need a direct summary field.
- Add a new contract constant for this single field. Rejected for now because
  the value is a simple projection of the existing structured gate.

## Risks

- Summary drift if the top-level field is hardcoded. Mitigation: derive it from
  the nested gate and assert JSON/Markdown parity.

## Test Plan

- Compile touched Python files.
- Run focused no-WING and Project Director tests.
- Run no-WING quick validation.
- Run the full pytest suite.
- Run memory lint and diff whitespace check.

## Where The Other Agent May Disagree

Kimi may prefer folding this into a broader summary-surface field registry.
That is reasonable later, but the narrow derived field is lower risk for the
current blocker rail.
