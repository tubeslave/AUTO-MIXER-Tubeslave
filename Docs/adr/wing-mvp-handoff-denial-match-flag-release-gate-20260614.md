# ADR: WING MVP Handoff Denial Match Flag Release Gate

Date: 2026-06-14
Status: Accepted

## Context

The no-WING handoff summary recomputes whether backend-port prestart denial
fields match the expected plan. The handoff packet also declares those match
flags. If the declared flags drift from the recomputed verdicts, the release
gate must fail closed.

## Options considered

1. Keep only recomputed summary flags.
2. Trust the handoff packet's declared flags.
3. Compare declared flags with recomputed truth and fail closed on mismatch.

## Decision

Use option 3. The release gate now emits a specific
`wing_readonly_preflight_handoff_packet_<field>_match_flag_mismatch` violation
for each stale backend-port prestart denial match flag.

## Why this won

This keeps stale or manually edited handoff packets from appearing current. It
also gives operators both declared and computed values in the no-WING summary.

## Rejected alternatives

- Summary-only recomputation: hides whether the packet itself is stale.
- Trusting declarations: unsafe for readiness gating.

## Implementation plan

- Capture declared `<field>_matches_plan` values from the packet.
- Compare them with recomputed packet-vs-plan verdicts.
- Preserve declared and computed values in summary JSON and markdown.
- Add a drift test that must fail closed.

## Test plan

- Focused no-WING release check tests.
- Focused WING handoff/no-WING/Project Director/rails tests.
- Compile, memory lint, full suite, and whitespace diff.

## Risks and rollback

Older packet JSON without the new declarations fails closed. Roll back by
removing the declared-vs-computed comparison if a migration path is needed, but
do not use stale packets for readiness closure.
