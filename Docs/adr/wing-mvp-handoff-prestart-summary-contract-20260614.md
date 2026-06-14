# ADR: WING MVP Handoff Prestart Summary Contract

## Context

The supervised WING MVP remains blocked on `WING read-only connection proof`.
The no-WING plan artifact already requires a passed backend-port prestart summary
with type `no_wing_release_check`, connection policy
`no_hardware_no_network_no_live_writes`, no backend/console contact, no test
execution, and a passed backend-port check status. The handoff packet previously
exposed the prestart summary path and SHA binding but not the full summary
contract.

## Options considered

1. Keep the contract only in the plan artifact.
2. Add the summary contract to the handoff packet without validation.
3. Add the summary contract to the handoff packet and make no-WING validation
   reject packet drift.

## Decision

Use option 3.

## Why this won

The handoff packet is the operator-facing pre-backend gate. It should be enough
to prove that the packet is bound to a passed no-WING prestart summary without
requiring the operator or validator to infer the boundary from the plan artifact
alone.

## Rejected alternatives

Keeping the contract only in the plan artifact leaves a stale-packet gap. Adding
fields without validation would improve reporting but not fail closed.

## Implementation plan

- Add prestart summary type/status/connection-policy/no-contact/check-status
  fields to the generated handoff packet.
- Render those fields in the packet Markdown.
- Compare packet values against the plan artifact and shared contract constants
  in the no-WING handoff validator.
- Include the fields in no-WING summary output.
- Add positive and negative tests.

## Test plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py --tb=short -x`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m py_compile tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py`
- Standard full pytest before commit.

## Risks and rollback

This is report-only offline validation and does not contact WING or grant live
authority. Roll back by removing the packet fields, validator checks, and tests
from this slice.
