# ADR: WING Handoff Structured Gate Summary

## Context

The WING read-only handoff packet now carries a structured
`supervised_prestart_gate`, and no-WING release validation rejects packets when
the structured gate does not match the final operator step. The release summary
still exposed only derived operator-sequence fields.

## Options considered

1. Keep summary output unchanged.
2. Add the structured gate and expected gate to JSON and Markdown summaries.
3. Replace existing operator-sequence summary fields with only the structured
   gate.

## Decision

Use option 2. The release summary now includes both `supervised_prestart_gate`
and `expected_supervised_prestart_gate`.

## Why this won

It gives reviewers the exact machine-readable gate while preserving existing
operator-sequence evidence. Keeping both surfaces makes mismatch debugging
straightforward.

## Rejected alternatives

Leaving summaries unchanged was rejected because it hid the new contract from
the main release artifacts. Replacing the older fields was rejected because they
remain useful for human review.

## Implementation plan

- Extend `wing_readonly_preflight_handoff_packet_artifact_summary`.
- Render the structured gate fields in Markdown output.
- Add regression assertions for JSON and Markdown summaries.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "summary_embeds_wing_preflight_handoff_packet_artifact or handoff_packet"`
- Affected WING/no-WING tests.
- `python3 tools/memory/memory_lint.py`
- Full pytest suite.

## Risks and rollback

The main risk is noisy Markdown. Roll back by replacing the JSON string line
with a more compact field list while keeping the JSON summary fields intact.
