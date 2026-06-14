# ADR: WING MVP Handoff Anchor Digest Review Step Index

## Context

The supervised WING MVP still waits on real WING read-only connection proof.
The no-WING handoff packet must keep supervised proof planning behind an
operator-reviewed current blocker anchor and digest. The previous gate validated
the sequence count and final supervised prestart gate index, but did not fail
closed when the anchor digest review step moved away from its expected index.

## Options considered

1. Keep the existing implicit index expectation in docs only.
2. Validate the step presence only.
3. Promote the expected anchor digest review step index into the shared
   preflight contract and fail closed on mismatch.

## Decision

Use option 3.

## Why this won

The handoff packet is the operator's no-WING checklist before real hardware
read-only work. A deterministic index makes reordered packets visible in JSON,
Markdown, Project Director text, and tests without adding live authority or
contacting hardware.

## Rejected alternatives

Docs-only wording does not protect validation. Presence-only validation still
allows a packet to keep the anchor review step but move it earlier in the
sequence.

## Implementation plan

Add a shared constant for the expected anchor digest review step index, expose
expected/match fields in no-WING detail and summary artifacts, add a negative
test for a shifted step that preserves count and final supervised gate index,
and update WING MVP/autoloop docs.

## Test plan

Run focused validator and rails tests, no-WING quick summary generation,
Project Director, direction guard, memory lint, `git diff --check`, and the
standard full test command.

## Risks and rollback

Risk is limited to stricter no-WING validation and documentation. Rollback is to
remove the new expected/match fields and mismatch violation if the operator
sequence contract changes.
