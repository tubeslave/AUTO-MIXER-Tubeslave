# ADR: WING MVP Handoff Declared Director Self-Check

## Context

The WING no-WING gate recomputes Project Director selected-action digest
fields and renders expected, actual, and match verdicts in the release summary.
The handoff packet itself only carried a shorter subset, which made the
operator-facing packet less complete before no-WING validation.

## Options considered

1. Keep the handoff packet short and rely only on no-WING-derived fields.
2. Add packet-declared expected/actual fields without validating them.
3. Add packet-declared fields and make no-WING validation compare them with the
   recomputed current Project Director directive.

## Decision

Use option 3.

## Why this won

The packet should be inspectable, but the no-WING gate must remain the
authority. Declaring the fields in the packet and then recomputing them during
validation gives both properties: clearer operator evidence and fail-closed
stale-packet detection.

## Rejected alternatives

No-WING-derived-only fields left the pre-validation packet under-specified.
Declared-only fields without validation would be misleading if the Project
Director directive changed.

## Implementation plan

Add packet fields for expected Project Director anchor/source/SHA, actual SHA,
and match verdicts. Render them in the handoff Markdown. Extend no-WING packet
validation to reject stale declared self-check fields and add focused positive
and negative tests.

## Test plan

Run focused handoff packet, no-WING, Project Director, and WING rail tests. Run
the packet builder, Project Director, Direction Guard, no-WING quick
fail-closed, memory lint, `git diff --check`, and the standard full test
command.

## Risks and rollback

Risk is limited to offline handoff metadata and validation. Rollback is to
remove the declared fields and validator comparisons if the packet schema is
replaced by a structured digest object. This change does not query or mutate a
console, update readiness, grant live authority, or close the blocker.
