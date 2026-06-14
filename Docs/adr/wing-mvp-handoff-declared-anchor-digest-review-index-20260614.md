# ADR: WING MVP Handoff Declared Anchor Digest Review Index

## Context

The supervised WING MVP still waits on real WING read-only connection proof.
The no-WING handoff packet now validates the actual anchor digest review step
index. The packet also declares the expected anchor digest review step index as
top-level metadata, but that declared value must not drift from the actual
operator sequence or the shared contract.

## Options considered

1. Keep the declared index informational only.
2. Remove the declared index from the handoff packet.
3. Validate the declared index against both the shared contract and the
   extracted operator-sequence index.

## Decision

Use option 3.

## Why this won

The handoff packet is an operator-facing no-WING artifact. Its metadata should
be as trustworthy as the extracted checklist. Validating the declared field
keeps stale or contradictory packets from looking safe before a hardware
session.

## Rejected alternatives

Informational-only metadata can become misleading. Removing the field would lose
a useful explicit review handle for the operator packet.

## Implementation plan

Add declared-index validation to no-WING release checks, expose declared
contract/operator-sequence match fields in JSON and Markdown summaries, render
the declared index in the handoff packet, and update WING MVP/autoloop docs.

## Test plan

Run focused validator and rails tests, no-WING quick summary generation,
Project Director, direction guard, memory lint, `git diff --check`, and the
standard full test command.

## Risks and rollback

Risk is limited to stricter offline handoff validation. Rollback is to remove
the declared-index mismatch checks and summary fields if the handoff packet
schema changes.
