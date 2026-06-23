# Proposal: WING MVP Packet Safety Boundary Summary JSON Surface

Date: 2026-06-15

## Short thesis

Promote the handoff packet safety-boundary fields into top-level no-WING
`summary.json` so automation can verify the non-executing read-only boundary
without parsing the nested packet artifact.

## What I understood

The active MVP remains supervised WING Rack proof, with `WING read-only
connection proof` still first. The latest no-WING summary exposes detailed
backend prestart surfaces, but core packet fields such as `read_only`,
`mutates_console_state`, `readiness_closure_allowed_by_packet`, and
`next_manual_artifact_required` remain nested.

## Proposed solution

Add a dedicated
`build_wing_readonly_packet_safety_boundary_summary_surface()` helper that
mirrors the safety-boundary fields from the handoff artifact into top-level
`summary.json`. Include a contract match verdict for
`next_manual_artifact_required` and explicit denial flags.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-packet-safety-boundary-summary-json-surface-20260615.md`
- `.ai/proposals/codex-wing-mvp-packet-safety-boundary-summary-json-surface-20260615.md`
- `Docs/adr/wing-mvp-packet-safety-boundary-summary-json-surface-20260615.md`
- `Docs/reports/AUTOMIXER_WING_MVP_PACKET_SAFETY_BOUNDARY_SUMMARY_JSON_SURFACE_20260615.md`
- `Docs/INDEX.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`

## Alternatives considered

Keep only the nested handoff artifact. This preserves existing behavior but
leaves the aggregate summary incomplete for first-command safety checks.

## Risks

The main risk is mistaking a complete safety boundary surface for readiness.
The surface must continue to deny execution, readiness closure, and live
authority.

## Test plan

- Focused no-WING summary test for packet safety-boundary fields.
- Quick no-WING release check with frontend skipped.
- Project Director JSON check.
- Focused WING MVP suite.
- Standard repository pytest command.
- Memory lint and git diff whitespace check.

## Where the other agent may disagree

Kimi may prefer a generic handoff packet projection. I would keep this scoped
because these fields define the non-executing packet boundary and are core to
the current first command.
