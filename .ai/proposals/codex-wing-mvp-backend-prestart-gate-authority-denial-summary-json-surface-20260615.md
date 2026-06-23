# Proposal: WING MVP Backend Prestart Gate Authority Denial Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the existing backend prestart gate authority-denial values and
packet-vs-plan match verdicts into top-level no-WING `summary.json` so the
Project Director current command can be audited without nested handoff parsing.

## What I understood

The active MVP remains supervised WING Rack proof, and `WING read-only
connection proof` is still the first blocker. The latest no-WING summary already
shows the backend prestart gate object at the top level, but raw denial fields
such as `backend_port_prestart_gate_execution_allowed` and
`plan_backend_port_prestart_gate_live_authority_granted` are still only inside
the handoff artifact.

## Proposed solution

Extend the existing backend prestart gate summary surface to mirror the
authority-denial fields and related match verdicts from the handoff artifact.
Add a dedicated summary-surface required marker and keep readiness/live
authority denied.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-backend-prestart-gate-authority-denial-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-backend-prestart-gate-authority-denial-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-backend-prestart-gate-authority-denial-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_AUTHORITY_DENIAL_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep only the nested handoff artifact. This avoids a small projection but keeps
the current first-command authority-denial checks harder to audit.

## Risks

The main risk is treating exposed denial fields as permission to proceed. The
surface must preserve the existing `false` authority values and add explicit
no-readiness/no-live-authority summary-surface flags.

## Test plan

- Focused no-WING summary test for top-level denial and match fields.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a generic handoff projection. I would keep this explicit
because the first-command contract names these authority-denial fields and the
existing file already uses dedicated summary-surface helpers.
