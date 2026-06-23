# ADR: WING MVP Supervised Prestart Step Text Summary Fields

## Context

The final supervised prestart step must stay aligned between the operator
sequence and the structured `supervised_prestart_gate`. A summary-level match
verdict makes that alignment directly reviewable before supervised proof
planning.

## Options Considered

1. Keep the step text only in nested/operator sequence fields.
2. Add top-level derived step text and match verdict fields.
3. Replace the structured gate step text with only a flat field.

## Decision

Add `supervised_prestart_gate_operator_sequence_step` and
`supervised_prestart_gate_operator_sequence_step_matches_summary` as derived
no-WING summary fields.

## Why This Won

The selected option improves reviewability without changing runtime behavior,
WING access, or live authority.

## Rejected Alternatives

- Nested-only validation is harder to grep.
- Flat-only validation loses structured handoff context.

## Implementation Plan

- Derive the step text from `supervised_prestart_gate`.
- Compare it to `operator_sequence_supervised_prestart_gate_step`.
- Render JSON/Markdown summary fields.
- Add tests and Project Director markers.

## Test Plan

- `PYTHONPATH=backend python -m py_compile ...`
- Focused pytest for no-WING, Project Director, handoff packet, supervised proof
  plan, and direction guard tests.
- no-WING quick release check.
- Full pytest suite.
- Memory lint and `git diff --check`.

## Risks And Rollback

Rollback is limited to derived fields and marker docs. Existing structured gate
and operator sequence fields remain intact.
