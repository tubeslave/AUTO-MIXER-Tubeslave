# ADR: WING MVP Handoff Capture Step Summary Surface

## Context

The no-WING handoff path must prove that the backend/WING connection is
confirmed before the operator opens the real read-only capture. Existing
validation computed this ordering inside the handoff artifact, but the flat
summary did not expose the capture step text, expected index, or match verdict.

## Options Considered

1. Keep the capture-step fields nested.
2. Add flat capture-step summary fields and match verdicts.
3. Replace the operator-sequence flat fields with a new structured object.

## Decision

Add flat no-WING summary fields for the read-only capture step:
`expected_operator_sequence_capture_readonly_preflight_step_index`,
`operator_sequence_capture_readonly_preflight_step_index_matches_contract`,
`operator_sequence_capture_readonly_preflight_step`,
`expected_operator_sequence_capture_readonly_preflight_step`, and
`operator_sequence_capture_readonly_preflight_step_matches_contract`.

## Why This Won

The operator can verify ordering and exact capture-step intent from
`summary.json` and `summary.md` before opening a backend/WING WebSocket. The
change does not contact hardware or alter live authority.

## Rejected Alternatives

- Nested-only fields require reviewers to inspect the artifact blob.
- A structured refactor is unnecessary for this narrow fail-closed gap.

## Implementation Plan

- Derive expected capture-step index and text from the shared preflight contract.
- Fail validation on capture-step index/text mismatch.
- Promote the fields to the flat no-WING summary and Markdown.
- Update tests, Project Director baseline markers, and runbook docs.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused no-WING and handoff packet tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to the flat capture-step fields and their explicit mismatch
checks. Existing handoff sequence hash checks remain intact.
