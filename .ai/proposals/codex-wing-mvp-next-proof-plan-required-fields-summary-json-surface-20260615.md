# Proposal: WING MVP Next Proof Plan Required Fields Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the next proof-plan required and missing supervised summary fields into
top-level no-WING `summary.json` so automation can verify the fail-closed
proof-plan contract without parsing the nested `next_wing_proof_plan`.

## What I understood

The active MVP remains supervised WING Rack proof. The nested
`next_wing_proof_plan` already exposes `required_supervised_proof_plan_summary_fields`
and `missing_required_supervised_proof_plan_summary_fields`, but the aggregate
summary leaves the missing-field result nested.

## Proposed solution

Add `build_next_wing_proof_plan_required_fields_summary_surface()` and merge it
when `next_wing_proof_plan` is present. The surface should expose required and
missing fields, the expected required-field contract, a match verdict, and
explicit readiness/live-authority denial flags.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-next-proof-plan-required-fields-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-next-proof-plan-required-fields-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-next-proof-plan-required-fields-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_NEXT_PROOF_PLAN_REQUIRED_FIELDS_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep the evidence nested under `next_wing_proof_plan`. That preserves behavior
but makes the aggregate artifact incomplete for first-command contract checks.

## Risks

The main risk is treating an empty missing-field list as live proof readiness.
The surface must continue to deny readiness closure and live authority; it only
describes summary-field completeness.

## Test plan

- Focused next-WING proof-plan summary test.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer keeping all next-plan evidence under `next_wing_proof_plan`.
I would still expose these fields because current automation treats the
aggregate `summary.json` as the handoff surface for first-command checks.
