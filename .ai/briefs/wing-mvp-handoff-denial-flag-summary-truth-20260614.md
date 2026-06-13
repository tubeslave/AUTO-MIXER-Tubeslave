# Brief: WING handoff denial flag summary truth

## Context

The WING read-only handoff summary is operator-facing audit evidence. It must
not make a failed or unsafe packet look safer by replacing packet-derived denial
or authority flags with hardcoded safe values.

## Task

Make the handoff summary preserve actual unsafe `true` values for packet
execution, readiness, live authority, and full-autonomous flags.

## Acceptance

- Failed unsafe handoff packets still fail validation.
- The summary artifact and Markdown preserve `true` values for unsafe packet
  flags instead of normalizing them to `false`.
- Existing safe handoff summaries still show those flags as `false`.
- No backend, WING, OSC, or live-write behavior is added.
