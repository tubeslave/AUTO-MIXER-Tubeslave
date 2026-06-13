# Brief: WING handoff backend prestart command summary

## Context

The WING read-only handoff summary is operator-facing audit evidence before a
real hardware session. The Project Director rail requires the no-WING handoff
summary to show the packet-declared `backend_port_prestart_check_command`
exactly, not only the command extracted from `commands.prestart_backend_port_check`.

## Task

Expose `backend_port_prestart_check_command` in the handoff summary artifact and
Markdown output.

## Acceptance

- Valid handoff summaries show the packet-declared backend prestart command.
- Failed handoff summaries preserve an unsafe or mismatched command value.
- The existing extracted `prestart_backend_port_check_command` remains visible.
- No backend, WING, OSC, or live-write behavior is added.
