# ADR: WING MVP handoff prestart summary SHA binding

## Context

The supervised WING MVP is still blocked on real WING read-only connection
proof. The no-WING prestart gate produces a summary file before any intended
backend/WING session. Later accepted-candidate review requires the real
preflight summary to bind that prestart summary by path and SHA256.

The handoff packet already carried the prestart summary path, but not the
digest. That left the operator-visible handoff weaker than the downstream
accepted-candidate contract.

## Options considered

1. Keep the digest only in the plan artifact.
2. Require handoff packets to include a digest for every prestart summary path.
3. Include a digest when the prestart summary file exists, and fail no-WING
   validation if it differs from the plan digest.

## Decision

Use option 3.

## Why this won

It tightens the real-session handoff without breaking placeholder-only packet
generation. A no-WING generated packet now carries the same prestart summary
digest as the plan artifact, while a standalone placeholder packet explicitly
shows that no digest is available.

## Rejected alternatives

Option 1 was weaker because the handoff packet is the operator-facing bridge
into the hardware step. Option 2 was too strict for offline packet previews that
still contain the documented placeholder.

## Implementation plan

- Hash the prestart summary in the handoff packet builder when the file exists.
- Add packet fields for the digest and digest availability.
- Make no-WING handoff artifact validation compare packet digest to plan digest.
- Render packet digest, plan digest, and match status in summary Markdown.

## Test plan

- Handoff packet builder tests for placeholder and existing-file paths.
- no-WING validator tests for matching and mismatched digest.
- Focused WING handoff/no-WING test suite.

## Risks and rollback

The change is read-only and does not contact WING. Rollback is reverting the
packet digest fields and validator comparison if they block legitimate no-WING
packets unexpectedly.
