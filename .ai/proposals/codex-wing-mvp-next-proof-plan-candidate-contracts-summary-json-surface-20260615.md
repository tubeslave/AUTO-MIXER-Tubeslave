# Proposal: WING MVP Next Proof Plan Candidate Contracts Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the already-validated `next_wing_proof_plan` candidate contract
verdicts into top-level no-WING `summary.json` fields so automation can verify
prestart and artifact-root gates without nested object parsing.

## What I understood

The active MVP remains supervised WING Rack proof, with `WING read-only
connection proof` still first. The no-WING proof-plan object already exposes
accepted-candidate prestart and artifact-root contract validity, violation
maps, candidate source/root lists, and invalid records. The aggregated summary
does not expose these values at the top level.

## Proposed solution

Add a narrow
`build_next_wing_proof_plan_candidate_contracts_summary_surface()` helper that
copies the relevant `next_wing_proof_plan` values into top-level
`summary.json`. Keep explicit readiness closure and live authority denial
fields.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-next-proof-plan-candidate-contracts-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-next-proof-plan-candidate-contracts-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-next-proof-plan-candidate-contracts-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_NEXT_PROOF_PLAN_CANDIDATE_CONTRACTS_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep only the nested `next_wing_proof_plan` object. This avoids a code change
but leaves the current first-command proof-plan contract harder for automation
to audit.

## Risks

The main risk is mistaking the projection for authority to proceed with
supervised proof. The new surface should only mirror existing no-WING values
and explicitly deny readiness closure and live authority.

## Test plan

- Focused no-WING summary test for top-level candidate contract values.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a generic top-level projection for all proof-plan fields. I
would keep this dedicated because the current gap is specific and the existing
release-check file already uses explicit summary-surface helpers.
