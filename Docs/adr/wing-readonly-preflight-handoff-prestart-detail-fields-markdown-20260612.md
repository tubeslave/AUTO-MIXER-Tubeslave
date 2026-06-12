# ADR: WING Read-Only Preflight Handoff Prestart Detail Fields Markdown

## Context

The shared WING read-only preflight contract and JSON handoff packet require
`backend_port_prestart_check_failure_detail_fields`, but the markdown handoff
packet did not render that list. The operator-facing artifact could therefore
hide the exact busy-port evidence fields required before starting the intended
backend/WING session.

## Options considered

1. Leave markdown unchanged and rely on JSON validators.
2. Render the existing field list in the markdown Safety Boundary.
3. Add separate prose that duplicates the field list.

## Decision

Use option 2. The markdown packet now renders the same
`backend_port_prestart_check_failure_detail_fields` list already present in the
packet payload.

## Why this won

The markdown handoff is the artifact an operator is most likely to read before
the hardware session. Rendering the existing machine-readable field list keeps
manual and validator surfaces aligned without adding new behavior.

## Rejected alternatives

Option 1 preserves operator-facing drift. Option 3 duplicates the list and would
be easier to let drift from the shared contract.

## Implementation plan

- Render `backend_port_prestart_check_failure_detail_fields` in
  `tools/build_wing_readonly_preflight_handoff_packet.py`.
- Extend `tests/test_wing_readonly_preflight_handoff_packet.py` to assert all
  shared field names are present in markdown output.
- Validate the WING/no-WING stack and full backend suite.

## Test plan

- Focused handoff markdown test.
- Adjacent WING/no-WING tests.
- Full backend suite.

## Risks and rollback

Risk: the handoff packet becomes slightly more verbose. Rollback: remove the
markdown line and test assertions. No hardware access or live write behavior is
changed.
