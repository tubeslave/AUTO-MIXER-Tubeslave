# Proposal: WING MVP Backend Prestart Gate Object Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the structured backend prestart gate objects from the handoff packet
artifact into top-level no-WING `summary.json` so automation can verify the
first operator-step gate without nested artifact parsing.

## What I understood

The active MVP remains supervised WING Rack proof, with `WING read-only
connection proof` still first. The Project Director current command requires
`backend_port_prestart_gate` to exactly match
`expected_backend_port_prestart_gate`, but the latest no-WING summary only
exposes the match verdict and command text at the top level.

## Proposed solution

Extend `build_backend_port_prestart_gate_summary_surface()` to mirror the
structured gate, expected gate, and declared expected gate objects. Keep the
existing no-authority summary-surface flags unchanged.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-backend-prestart-gate-object-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-backend-prestart-gate-object-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-backend-prestart-gate-object-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_OBJECT_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep only match verdicts. This is compact but leaves the current first-command
gate object hidden inside the handoff artifact.

## Risks

The main risk is readers mistaking an exposed gate object for permission to
start a WING session. The object itself and the summary-surface flags must keep
live authority denied.

## Test plan

- Focused no-WING summary test for top-level backend prestart gate objects.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer exposing every handoff gate object in one generic projection.
I would keep this scoped to the backend prestart gate because it is the first
operator step and the current Project Director directive names this exact
object.
