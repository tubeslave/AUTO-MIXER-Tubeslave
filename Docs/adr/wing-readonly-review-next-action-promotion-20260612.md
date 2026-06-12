# ADR: WING read-only review next action promotion

## Context

The WING read-only review report can include invalid or blocked artifacts.
Those artifacts may contain an `operator_next_step`, but the report-level
`next_required_action` previously stayed generic.

## Options Considered

1. Keep next-step guidance only in artifact details.
2. Add a separate diagnostics section only.
3. Promote artifact next steps into the primary report next action when blocked.

## Decision

Promote artifact next steps into `next_required_action` when no clean candidate
is available, and render all unique artifact next steps in Markdown.

## Why This Won

The report is meant for manual operator review. A blocked expected-plan or
prestart-contract failure should tell the operator exactly what to regenerate
before any backend/WING WebSocket is opened.

## Rejected Alternatives

Keeping the guidance only in artifact details was rejected because it is easy
to miss. A diagnostics-only section was rejected because it would duplicate
information while leaving the primary next action generic.

## Implementation Plan

- Collect unique `operator_next_step` values from artifact rows.
- Add `artifact_operator_next_steps` to the report.
- Use the first step as `next_required_action` for blocked reports.
- Add a regression for blocked expected-plan repair guidance.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_evidence.py`
- Adjacent WING/no-WING tests.
- Full test suite before commit.

## Risks And Rollback

The guidance must not be interpreted as live authority. Rollback is to remove
the promotion and keep the artifact-row field only.
