# Proposal: WING MVP Backend Prestart Failure Detail Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the backend prestart failure-detail field list into top-level no-WING
`summary.json` so operators and automation can verify the fail-closed listener
review contract without parsing the nested handoff artifact.

## What I understood

The active MVP remains supervised WING Rack proof, with `WING read-only
connection proof` still first. The latest no-WING summary exposes backend gate
objects and authority denials, but the canonical
`backend_port_prestart_check_failure_detail_fields` list is still nested inside
the handoff artifact.

## Proposed solution

Extend `build_backend_port_prestart_gate_summary_surface()` to expose the
failure-detail field list, expected contract list, and a match verdict. Keep
explicit readiness-closure and live-authority denial flags.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-backend-prestart-failure-detail-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-backend-prestart-failure-detail-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-backend-prestart-failure-detail-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_FAILURE_DETAIL_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep the list nested. This preserves existing behavior but keeps an important
fail-closed operator review contract harder to audit from the aggregate
summary.

## Risks

The main risk is confusing a complete failure-detail list with permission to
start a WING session. The summary surface must keep readiness closure and live
authority denied.

## Test plan

- Focused no-WING summary test for top-level failure-detail fields.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a broader handoff-artifact projection. I would keep this
scoped because the current first-command contract specifically names the
failure-detail list.
