# ADR: WING MVP Handoff Gate Match Verdicts

## Context

The WING no-WING handoff validator already compared backend and supervised
prestart gates with their expected contract objects. The handoff packet and
summary did not expose named match verdict fields for those comparisons, which
made the operator-facing proof less explicit than other recent packet
self-checks.

## Options considered

1. Keep only object equality validation.
2. Add named packet flags without validating them.
3. Add named packet flags and make no-WING recompute and validate them.

## Decision

Use option 3.

## Why this won

It preserves the existing object equality checks while making the verdicts
operator-visible and machine-readable. The no-WING gate stays authoritative and
rejects stale declared flags.

## Rejected alternatives

Object-equality-only checks were correct but less inspectable. Declared-only
flags could become stale if the expected gate contract changes.

## Implementation plan

Add named backend and supervised gate match fields to the packet. Recompute the
same booleans in no-WING validation, reject stale declarations, expose them in
the no-WING summary and Markdown, and update Project Director/rail markers and
focused tests.

## Test plan

Run focused handoff, no-WING, Project Director, and WING rail tests. Run packet
generation, Direction Guard, no-WING quick fail-closed, memory lint,
`git diff --check`, and the standard full test command.

## Risks and rollback

Risk is limited to offline handoff metadata and validation. Rollback is to
remove the new named verdict fields and rail markers if the packet schema moves
to a structured gate-verdict object. This does not query or mutate a console,
update readiness, grant live authority, or close the blocker.
