# Brief: WING MVP Operator Markdown Required Field Lines Self-Check

Date: 2026-06-15
Status: active implementation brief

## Task

Make the no-WING operator Markdown surface fail closed when the
`operator_markdown_summary_required_field_lines` list is stale or omits any
field required by the current proof-plan Markdown contract.

## Context

The WING MVP remains blocked on real read-only connection proof. The no-WING
operator `summary.md` now requires value-gate authority-denial lines, but the
validator also needs to prove that the rendered required-field-lines list names
the whole current contract.

## Safety Boundary

No WING session, backend/WING connection, hardware query, or live write is
allowed. The change is limited to no-WING local validation and documentation.

## Deliverables

- Runtime self-check for the rendered required-field-lines list.
- Unit coverage for stale required-field-lines lists.
- Current First Command, Project Director, MVP, runbook, ADR, and report sync.

## Test Plan

- Focused no-WING and Project Director tests.
- no-WING quick with frontend skipped.
- Full pytest and hygiene checks if focused validation passes.
