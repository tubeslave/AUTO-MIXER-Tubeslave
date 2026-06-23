# ADR: WING MVP Required Supervised Prestart Marker Verdict Summary Fields

## Context

The no-WING handoff summary exposes both the top-level required supervised
prestart marker list and the structured supervised gate marker list. The
structured gate list has direct match verdicts, but the top-level list did not.

## Options Considered

1. Keep top-level required marker-list integrity implicit.
2. Add top-level required marker-list match verdicts.
3. Replace list verdicts with count-only checks.

## Decision

Add top-level required marker-list match verdicts while preserving the existing
structured gate and count checks.

## Why This Won

It makes the top-level marker-list contract directly reviewable without
weakening the stricter structured gate checks. It does not contact WING
hardware, grant live authority, or close readiness.

## Rejected Alternatives

- Implicit validation forces reviewers to infer one side of the check.
- Count-only validation is not strong enough for a live-sound safety rail.

## Implementation Plan

- Derive `required_supervised_prestart_gate_markers_match_contract`.
- Derive `required_supervised_prestart_gate_markers_match_supervised_gate`.
- Render JSON/Markdown fields.
- Add tests and Project Director baseline markers.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to the derived verdict fields and docs. Existing marker and
count checks remain intact.
