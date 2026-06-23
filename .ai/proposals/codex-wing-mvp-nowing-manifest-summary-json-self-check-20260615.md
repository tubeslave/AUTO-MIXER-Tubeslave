# Proposal: no-WING Manifest Summary JSON Self-Check

## Short thesis

Make the manifest summary surface self-check symmetric: JSON fields and
Markdown lines are both verified and fail closed.

## What I understood

The WING supervised MVP remains blocked on a real read-only preflight. The
offline no-WING summaries are operator gates, so their self-reported
`missing_fields: []` values must be computed, not assumed.

## Proposed solution

- Add a helper that computes missing manifest summary JSON fields.
- Call it before rendering `summary.md`.
- Keep the existing Markdown line check as a separate surface check.
- Add a focused test for missing JSON fields.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`

## Alternatives considered

- Leave the field as a hardcoded empty list: rejected because stale summary
  JSON could claim a complete surface.
- Fold JSON and Markdown checks into one function: rejected because the two
  surfaces have different failure evidence.

## Risks

- Future required fields must still be added to the summary renderer.
- This does not prove real WING readiness; it only tightens offline reporting.

## Test plan

- Focused no-WING manifest summary tests.
- no-WING quick check.
- Project Director check.
- Standard pytest if focused checks are clean.

## Where the other agent may disagree

Kimi may prefer a schema object for all no-WING summary fields. That is a
reasonable later refactor, but the smaller helper closes the current stale
summary-field risk without broad churn.
