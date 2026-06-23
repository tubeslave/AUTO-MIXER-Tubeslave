# Brief: WING MVP Operator Markdown Value-Gate Authority Denial

Date: 2026-06-15
Status: active implementation brief

## Task

Keep the WING read-only connection proof gate moving without granting live
authority by making the no-WING operator Markdown contract require the
accepted-candidate value-gate authority-denial lines.

## Context

The current first blocker remains `WING read-only connection proof`. The
no-WING `summary.md` already renders the accepted-candidate value-gate
summary-surface denial fields, but the machine-readable
`operator_markdown_summary_required_field_lines` contract did not require those
denial lines explicitly.

## Safety Boundary

This task must not contact WING, start a backend/WING session, query hardware,
or grant live authority. The expected authority verdicts remain
`readiness_closure_allowed: false` and `live_authority_granted: false`.

## Deliverables

- Extend the no-WING operator Markdown required field-line contract.
- Cover the new required denial lines in tests.
- Sync Current First Command and operator docs with the contract.
- Record the decision and validation evidence.

## Test Plan

- Focused no-WING tests.
- Project Director baseline coverage tests.
- no-WING quick check with frontend skipped.
- Full pytest if runtime remains stable.
