# Proposal: WING MVP Operator Sequence Step Gate Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the already-validated handoff operator-sequence step and supervised
prestart gate values into top-level no-WING `summary.json` fields so automation
can verify the current first command without nested artifact parsing.

## What I understood

The active MVP remains supervised WING Rack proof, with `WING read-only
connection proof` still first. The no-WING handoff artifact already validates
the anchor digest review step at index 8 and the supervised prestart gate at
final index 9, but the aggregated summary leaves those top-level fields empty.

## Proposed solution

Add a narrow `build_wing_readonly_operator_sequence_step_gate_summary_surface()`
helper that copies step-index, gate-marker, and supervised-prestart-gate
verdict fields from the handoff packet artifact into top-level `summary.json`.
Keep explicit readiness closure and live authority denial fields.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-operator-sequence-step-gate-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-operator-sequence-step-gate-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-operator-sequence-step-gate-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_STEP_GATE_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep only the nested handoff packet artifact. This avoids code churn but keeps
the current first-command evidence harder for automation to audit. A generic
projection layer would be larger than this specific safety/reporting gap.

## Risks

The main risk is misreading a reporting projection as readiness authority. The
new surface should only mirror already-validated values and explicitly deny
readiness closure and live authority.

## Test plan

- Focused no-WING summary test for top-level step-gate fields.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may argue this belongs in a broader handoff artifact projection contract.
I would keep this dedicated helper because the existing release-check file
already uses dedicated summary-surface helpers and the requested evidence is
specific.
