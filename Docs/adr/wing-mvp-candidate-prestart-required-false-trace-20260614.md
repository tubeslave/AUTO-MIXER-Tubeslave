# ADR: WING MVP candidate prestart required-false trace

## Context

The WING read-only evidence intake validates the referenced no-WING prestart
backend-port detail contract, including required-false fields such as
`safe_to_stop_listener_by_gate` and prestart gate authority flags. Downstream
candidate review rows did not preserve those fields for decision review and
proof-plan gating.

## Options considered

1. Keep validation only in evidence intake.
2. Render the fields only in markdown.
3. Make the fields part of the shared accepted-candidate prestart contract.

## Decision

Use option 3. The required-false detail fields are now carried through evidence
records, artifact review rows, decision candidate rows, and the shared
prestart-contract helper.

## Why this won

The future supervised proof plan relies on accepted source ids and candidate
review rows. The contract should fail closed if those rows omit or invert the
prestart safety denial fields.

## Rejected alternatives

Markdown-only rendering was rejected because it would not protect programmatic
proof-plan gating. Evidence-only validation was rejected because later summaries
would still be weaker than the accepted source.

## Implementation plan

- Store the fields on `WingReadonlyPreflightArtifactRecord`.
- Copy them into artifact and decision review rows.
- Require each field to be exactly `false` in the shared helper.
- Cover valid and unsafe candidate rows with focused tests.

## Test plan

- `tests/test_wing_readonly_preflight_evidence.py`
- `tests/test_wing_readonly_preflight_review_tool.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision_tool.py`

## Risks and rollback

Risk: stale external candidate summaries may newly fail closed. Roll back by
removing the helper requirement and record/report fields, but keep evidence
intake validation if post-run artifact rejection is still desired.
