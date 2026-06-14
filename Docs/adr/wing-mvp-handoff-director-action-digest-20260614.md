# ADR: WING MVP Handoff Director Action Digest

## Context

The supervised WING MVP is still blocked on real WING read-only connection
proof. Project Director now emits a selected-action digest and current-first-
command baseline coverage verdict, but the no-WING handoff packet did not carry
that digest. A packet could therefore be valid for the blocker anchor and
operator sequence while not proving it matched the exact Project Director
selected action.

## Options considered

1. Leave the digest only in Project Director output.
2. Add a handoff packet digest field without no-WING validation.
3. Add a Project Director selected-action anchor and validate it in no-WING.

## Decision

Use option 3.

## Why this won

The handoff packet is the operator-facing bridge into the hardware step. It
should prove the selected first-blocker action it was built from, not merely
trust adjacent Project Director output. Validating the anchor and SHA256 keeps
the packet auditable without granting any live authority.

## Rejected alternatives

Keeping the digest outside the packet leaves a drift gap. Adding an unvalidated
field creates another stale metadata risk. Importing the full Project Director
agent into no-WING validation was rejected because it would pull direction-guard
evidence collection into a lightweight packet schema check.

## Implementation plan

Add a shared no-WING helper that reuses `update_mvp_progress` to build the
current Project Director selected-action anchor. Emit the anchor, source,
SHA256, availability flag, and current-first-command baseline coverage fields
from the handoff builder. Validate all of those fields in no-WING release checks
and expose them in JSON and Markdown summaries. Add regression tests for the
happy path and stale digest failure.

## Test plan

Run focused handoff, no-WING schema, and Project Director tests. Then run
Project Director, direction guard, no-WING quick fail-closed, memory lint,
`git diff --check`, and the standard full test command.

## Risks and rollback

Risk is limited to stricter offline handoff validation. Rollback is to remove
the Project Director selected-action anchor fields and mismatch violations if
the packet schema changes. This change does not start WING, query a console,
write to a console, update readiness, or close the first blocker.
