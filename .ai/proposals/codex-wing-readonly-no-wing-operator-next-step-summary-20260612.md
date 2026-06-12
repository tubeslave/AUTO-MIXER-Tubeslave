# Proposal: Preserve WING operator next steps in no-WING summaries

## Short Thesis

The no-WING release summary should preserve artifact-level
`operator_next_step` values from WING read-only evidence so blocked repair
guidance survives every offline review surface.

## What I Understood

The first MVP blocker still waits for real WING read-only proof. Offline
checks should make the preflight path safer and clearer, but must not treat
offline evidence as readiness proof.

## Proposed Solution

- Collect unique `operator_next_step` values from WING evidence artifacts in
  `validate_wing_readonly_preflight_evidence_snapshot`.
- Include them in `wing_readonly_preflight_evidence_summary`.
- Render them in no-WING Markdown.
- Add tests for duplicate de-duplication and summary rendering.

## Likely Files To Touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## Alternatives Considered

Leaving the guidance only in review reports was rejected because no-WING
summaries are the operator's prestart gate before the intended WING session.

## Risks

The guidance must remain diagnostic only. Existing invalid-artifact failures
and no-live-authority assertions stay unchanged.

## Test Plan

- Focused no-WING WING-preflight tests.
- Adjacent WING/no-WING suite.
- Full pytest before commit.

## Where The Other Agent May Disagree

Kimi may argue this duplicates review-report data. I prefer duplication here
because the no-WING summary is the prestart artifact operators scan first.
