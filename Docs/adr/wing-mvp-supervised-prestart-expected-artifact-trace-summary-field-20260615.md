# ADR: WING MVP Supervised Prestart Expected Artifact Trace Summary Field

## Context

The WING supervised proof path must stay behind an accepted read-only candidate
with a complete expected-artifact trace. The structured
`supervised_prestart_gate` already records this requirement, but top-level
no-WING summaries did not expose the specific expected-artifact trace gate.

## Options Considered

1. Keep only the nested structured gate.
2. Add a top-level derived field in no-WING JSON and Markdown summaries.
3. Replace the structured gate with only flat fields.

## Decision

Add the top-level derived field
`supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace`
while preserving the structured gate.

## Why This Won

The derived field keeps the detailed contract intact and adds a direct
operator/automation check surface. It does not grant live authority, execute
tests, query a console, or close the WING blocker.

## Rejected Alternatives

- Nested-only validation leaves the gate harder to audit from release summaries.
- Flat-only validation would remove useful structured context from the handoff
  packet.

## Implementation Plan

- Derive the field from `supervised_prestart_gate`.
- Publish it in `summary.json` and `summary.md`.
- Add tests that compare nested artifact state, top-level summary state, and
  Markdown rendering.
- Add the marker to Project Director baseline coverage and docs.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and git diff whitespace check.

## Risks And Rollback

If the field causes summary-surface drift, rollback is limited to the derived
field, docs, and focused assertions. The nested structured gate remains the
source of truth.
