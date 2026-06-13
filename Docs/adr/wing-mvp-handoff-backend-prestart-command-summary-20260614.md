# ADR: WING handoff backend prestart command summary

## Context

The WING read-only handoff summary is part of the operator audit trail before a
real hardware session. The Project Director rail requires the no-WING handoff
summary to expose `backend_port_prestart_check_command` exactly matching
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`.

## Options considered

1. Keep showing only `prestart_backend_port_check_command`.
2. Add the packet-declared `backend_port_prestart_check_command` beside the
   extracted command.
3. Rename `prestart_backend_port_check_command` to the packet field name.

## Decision

Use option 2. The summary must show both the packet-declared command and the
command extracted from the handoff command map.

## Why this won

It satisfies the exact rail wording without removing existing evidence. A
failed packet with a mismatched top-level command remains auditable in the JSON
and Markdown summaries.

## Rejected alternatives

Showing only the extracted command leaves the packet-declared field invisible.
Renaming the existing field creates unnecessary churn and loses the distinction
between packet and command-map sources.

## Implementation plan

- Add the field to `wing_readonly_preflight_handoff_packet_artifact_summary`.
- Render the field in `write_summary`.
- Extend safe and unsafe handoff summary tests.
- Add an operator-facing report and blocker evidence entry.

## Test plan

- Focused no-WING handoff tests.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Risks and rollback

The only behavioral change is extra summary visibility. Rollback would remove
the field and restore the previous weaker handoff summary.
