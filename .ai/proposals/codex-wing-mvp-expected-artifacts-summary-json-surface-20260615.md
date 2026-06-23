# Proposal: WING MVP Expected Artifacts Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the handoff packet expected-artifact contract into top-level no-WING
`summary.json` so automation can verify the real WING artifact targets without
parsing the nested packet artifact.

## What I understood

The active MVP remains supervised WING Rack proof. The current first command
requires the no-WING handoff summary to show expected artifacts and match
verdicts, but latest aggregate `summary.json` left those values nested.

## Proposed solution

Add `build_wing_readonly_expected_artifacts_summary_surface()` that mirrors
`expected_artifacts`, `expected_artifact_globs`, `plan_expected_artifacts`,
`expected_artifacts_match_plan`, summary/event glob strings, and match verdicts
into top-level `summary.json`. Keep explicit readiness/live-authority denial
flags.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-expected-artifacts-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-expected-artifacts-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-expected-artifacts-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_EXPECTED_ARTIFACTS_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep expected-artifact evidence nested under the handoff artifact. That
preserves behavior but leaves the aggregate artifact incomplete for
first-command checks.

## Risks

The main risk is treating artifact-target completeness as readiness. The
surface must continue to deny readiness closure and live authority; it only
describes the expected output contract.

## Test plan

- Focused no-WING handoff summary test.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer keeping all handoff values nested. I would expose these fields
because the current first command names them directly as handoff summary
requirements.
