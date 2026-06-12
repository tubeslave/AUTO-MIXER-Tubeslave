# ADR: WING Handoff Structured Prestart Gate

## Context

The WING read-only preflight handoff packet already ends with a final
operator-sequence step that requires `accepted_decision_ready`, a valid
accepted-candidate prestart trace, and
`WING_SUPERVISED_FADER_GAIN_PROOF`. Release validation computed this gate from
the prose step, but the handoff packet did not expose the same requirement as a
machine-readable contract.

## Options considered

1. Keep validating only the operator-sequence prose.
2. Add a structured `supervised_prestart_gate` object and validate it against
   the final operator-sequence step.
3. Replace the operator-sequence step with only structured JSON.

## Decision

Use option 2. The packet now includes `supervised_prestart_gate` with the final
step index, required markers, runbook target, accepted decision status,
accepted-candidate prestart trace requirement, and `grants_live_authority:
false`.

## Why this won

It preserves operator-readable instructions while giving automation an explicit
contract. The no-WING validator fails closed if the structured block is missing
or diverges from the final operator step.

## Rejected alternatives

Prose-only validation was rejected because it is harder for future automation to
consume safely. JSON-only was rejected because the handoff remains an operator
artifact.

## Implementation plan

- Add the structured gate to the handoff generator.
- Render it in handoff Markdown.
- Validate the block in `tools/no_wing_release_check.py`.
- Add generator and validator regressions.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k handoff_packet`
- Affected WING/no-WING tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is over-specific validation if the handoff sequence changes. Roll
back by updating both the structured contract and final operator step together,
then adjusting the validator expectation in the same commit.
