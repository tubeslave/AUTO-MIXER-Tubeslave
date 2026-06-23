# Proposal: WING MVP Operator Sequence Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the already-validated handoff operator-sequence contract values into
top-level no-WING `summary.json` fields so automation can verify the checklist
without parsing the nested packet artifact.

## What I understood

The active MVP remains supervised WING Rack proof, with `WING read-only
connection proof` still first. The current no-WING handoff artifact already
validates operator sequence count, sequence digest, expected digest, and
declared contract flags, but `summary.json` leaves the same top-level fields
empty.

## Proposed solution

Add a narrow `build_wing_readonly_operator_sequence_summary_surface()` helper
that copies operator-sequence fields from the handoff packet artifact into the
top-level summary. Add explicit `readiness_closure_allowed: false` and
`live_authority_granted: false` surface flags.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-operator-sequence-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-operator-sequence-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-operator-sequence-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep consumers parsing the nested handoff packet artifact. This avoids a code
change but preserves a brittle machine-consumer path and leaves the current
first command harder to audit.

## Risks

The main risk is confusing an offline summary projection for readiness
authority. The helper should only mirror validated values and should explicitly
deny readiness closure and live authority.

## Test plan

- Focused no-WING summary test for top-level operator-sequence fields.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer adding a generic projection helper for all nested handoff
fields. I would keep this scoped because the current gap is specific and the
existing file already uses dedicated summary-surface helpers.
