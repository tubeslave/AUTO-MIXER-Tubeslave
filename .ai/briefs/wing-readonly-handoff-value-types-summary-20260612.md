# Brief: WING read-only handoff value types summary

## Context

The WING read-only plan and capture commands already use
`--value-types fader,gain,pan`. The handoff command could rely on the handoff
tool default, but the no-WING summary did not surface the handoff value types or
prove they match the capture command.

## Task

Make the handoff command and no-WING summary explicitly carry and validate the
same read-only value-type set.

## Acceptance

- The handoff command includes `--value-types fader,gain,pan`.
- no-WING evidence reports `handoff_packet_value_types`.
- no-WING evidence fails closed if handoff value types do not match capture
  value types.
- No WING/backend/OSC contact is introduced.
