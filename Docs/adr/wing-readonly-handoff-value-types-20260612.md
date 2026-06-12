# ADR: WING read-only handoff value types

## Context

The read-only preflight runner accepts only `fader`, `gain`, and `pan` value
types. It also rejects empty and duplicated value-type lists. The handoff packet
generator used the same default values but did not reject invalid overrides.

## Options Considered

1. Keep relying on the runner to fail closed.
2. Add the same fail-closed validation to the offline handoff packet generator.

## Decision

Validate handoff value types before writing a packet.

## Why This Won

The handoff packet is the operator-facing bridge into the real hardware step.
It should not produce a packet that is already known to be invalid for the
runner's read-only contract.

## Rejected Alternatives

Relying only on runner validation was rejected because it delays a deterministic
input error until hardware-session prep.

## Implementation Plan

- Add value-type validation to `tools/build_wing_readonly_preflight_handoff_packet.py`.
- Add focused tests for empty, duplicated, and unsupported values.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py`
- Adjacent WING/no-WING tests.

## Risks And Rollback

The change is offline and does not contact WING, backend WebSocket, or OSC.
Rollback is to remove the handoff validation and rely on runner validation.
