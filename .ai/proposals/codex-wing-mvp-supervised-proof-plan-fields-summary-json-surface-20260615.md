# Proposal: WING MVP Supervised Proof Plan Fields Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the already-validated handoff supervised proof-plan field contract into
top-level no-WING `summary.json` fields so automation can verify the current
first command without nested artifact parsing.

## What I understood

The active MVP remains supervised WING Rack proof, with `WING read-only
connection proof` still first. The no-WING handoff artifact already validates
the required supervised proof-plan summary field list, expected list, declared
list, match verdicts, and invalid-record inclusion. The aggregated summary
leaves those top-level fields empty.

## Proposed solution

Add a narrow
`build_wing_readonly_supervised_proof_plan_fields_summary_surface()` helper
that copies the required field lists and verdicts from the handoff packet
artifact into top-level `summary.json`. Keep explicit readiness closure and
live authority denial fields.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-supervised-proof-plan-fields-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-supervised-proof-plan-fields-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-supervised-proof-plan-fields-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_PROOF_PLAN_FIELDS_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep only the nested handoff packet artifact. This avoids code churn but keeps
the proof-plan field contract harder for automation to audit. A generic
projection framework would be larger than this specific reporting gap.

## Risks

The main risk is mistaking the reporting projection for proof-plan authority.
The new surface should only mirror already-validated values and explicitly deny
readiness closure and live authority.

## Test plan

- Focused no-WING summary test for top-level proof-plan field contract values.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a generic projection layer for handoff fields. I would keep a
dedicated helper because the existing release-check file already uses explicit
summary-surface helpers and the requested proof-plan field contract is narrow.
