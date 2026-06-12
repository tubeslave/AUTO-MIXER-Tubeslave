# Brief: WING read-only handoff value types

## Context

The WING read-only runner fails closed when `--value-types` is empty,
duplicated, or outside `fader,gain,pan`. The offline handoff packet generator
accepted those invalid values and could write an operator packet that the
runner would later reject.

## Task

Make the handoff packet generator validate value types before writing a packet.

## Acceptance

- Empty value types are rejected.
- Duplicate value types are rejected.
- Unsupported value types are rejected.
- No WING/backend/OSC contact is introduced.
