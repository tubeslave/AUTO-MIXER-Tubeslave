# ADR: WING MVP Supervised Prestart Step Index Summary Fields

## Context

The supervised proof path must only unlock after the final operator sequence
gate. The structured `supervised_prestart_gate` captures the requirement, step
index, and final-step flag, but release summaries should expose those values
directly.

## Options Considered

1. Keep required/final-step values only inside the structured gate.
2. Mirror required/final-step values as top-level no-WING summary fields.
3. Replace the structured gate with flat fields only.

## Decision

Mirror `required`, `operator_sequence_step_index`, and
`operator_sequence_step_must_be_final` as top-level derived no-WING summary
fields while preserving the structured gate.

## Why This Won

The chosen option improves operator review and Project Director baseline
coverage without widening live authority or changing WING behavior.

## Rejected Alternatives

- Nested-only validation is harder to grep.
- Flat-only validation would remove useful structured handoff context.

## Implementation Plan

- Derive fields from `supervised_prestart_gate`.
- Render them in no-WING JSON and Markdown summaries.
- Add focused tests for artifact, top-level summary, and Markdown parity.
- Add Project Director baseline markers and documentation.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to derived summary fields and marker docs. The structured
gate remains source of truth.
