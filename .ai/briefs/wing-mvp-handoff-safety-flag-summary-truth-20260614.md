# Brief: WING handoff safety flag summary truth

## Context

The WING read-only handoff summary is operator-facing audit evidence before a
real hardware session. It must not make a failed packet look read-only or
non-executing by replacing packet-derived safety flags with hardcoded safe
values.

## Task

Make the handoff summary preserve actual unsafe `true` values for packet
`live_write`, `mutates_console_state`, `queries_console`, and `executes_tests`
fields.

## Acceptance

- Failed unsafe handoff packets still fail validation.
- The summary artifact and Markdown preserve unsafe `true` values for safety
  flags instead of normalizing them to `false`.
- Existing safe handoff summaries still show those flags as `false`.
- No backend, WING, OSC, test execution from handoff, or live-write behavior is
  added.
