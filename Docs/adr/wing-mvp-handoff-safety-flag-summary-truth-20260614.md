# ADR: WING handoff safety flag summary truth

## Context

The WING read-only handoff summary is part of the operator audit trail before a
real hardware session. A failed packet with unsafe `true` values for
`live_write`, `mutates_console_state`, `queries_console`, or `executes_tests`
must not be rendered as if those fields were safely false.

## Options considered

1. Keep hardcoded false values in the summary and rely on violations.
2. Preserve actual unsafe boolean values in the existing summary fields.
3. Add separate raw fields while keeping the existing fields normalized.

## Decision

Use option 2. The existing summary fields must reflect unsafe `true` values
from the packet detail while normal safe packets continue to show `false`.

## Why this won

The operator-facing summary should be truthful even when validation fails.
Preserving unsafe values makes stale or edited handoff packets easier to
diagnose before any WING interaction.

## Rejected alternatives

Hardcoded safe values hide useful failure context. Separate raw fields add
schema noise and still leave the main fields misleading.

## Implementation plan

- Update `wing_readonly_preflight_handoff_packet_artifact_summary`.
- Extend unsafe handoff tests to assert JSON and Markdown summary values.
- Add an operator-facing report and blocker evidence entry.

## Test plan

- Focused no-WING handoff tests.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Risks and rollback

Failed unsafe summaries may now show explicit `true` safety flags. That is the
desired audit behavior. Rollback would restore hardcoded safe values and weaken
failure diagnostics.
