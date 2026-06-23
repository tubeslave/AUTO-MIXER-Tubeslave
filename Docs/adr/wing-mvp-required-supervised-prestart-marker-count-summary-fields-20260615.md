# ADR: WING MVP Required Supervised Prestart Marker Count Summary Fields

## Context

The no-WING handoff summary compares the structured supervised gate marker count
to the top-level required marker list. The structured count is visible, but the
top-level required marker count was implicit.

## Options Considered

1. Keep the top-level required marker count implicit.
2. Add top-level required marker count and count-match verdicts.
3. Replace marker-list equality checks with count-only checks.

## Decision

Add top-level required marker count fields while preserving all existing marker
list equality checks.

## Why This Won

It makes both sides of the count comparison directly auditable and keeps the
stricter list checks intact. It does not contact WING hardware, grant live
authority, or close readiness.

## Rejected Alternatives

- Implicit count validation forces operators to infer one side of the check.
- Count-only validation is insufficient for a live-sound safety gate.

## Implementation Plan

- Derive the required marker count from
  `required_supervised_prestart_gate_markers`.
- Derive the expected count from the shared supervised proof marker contract.
- Compare the required count to the contract and structured supervised gate
  count.
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

Rollback is limited to the derived count fields and docs. Existing marker-list
checks remain intact.
