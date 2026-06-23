# Proposal: WING MVP Blocker Anchor Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the handoff packet MVP blocker anchor into top-level no-WING
`summary.json` so the Project Director first command can verify the current
first open blocker without parsing the nested packet artifact.

## What I understood

The active MVP remains supervised WING Rack proof, and `WING read-only
connection proof` is still the first open blocker. The latest no-WING summary
contains the anchor only under `wing_readonly_preflight_handoff_packet_artifact`,
while the current first command asks for top-level anchor and digest evidence.

## Proposed solution

Add `build_wing_readonly_mvp_blocker_anchor_summary_surface()` that mirrors the
nested `mvp_blocker_anchor`, SHA256 fields, current-first-open verdicts, and
the no-close marker into top-level `summary.json`. Keep explicit readiness and
live-authority denial flags on the surface.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-blocker-anchor-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-blocker-anchor-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-blocker-anchor-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_ANCHOR_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep only the nested handoff artifact. This avoids duplication but leaves the
aggregate summary incomplete for the current first-command contract.

## Risks

The main risk is readers mistaking anchor presence for proof completion. The
surface must keep `packet_does_not_close_blocker: true` and explicit no-live
authority flags.

## Test plan

- Focused no-WING summary test for blocker-anchor fields.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a generic projection of every nested handoff field. I would
keep this scoped because the current blocker only needs anchor identity,
digest, and no-close evidence.
