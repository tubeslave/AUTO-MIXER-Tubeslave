# Read-only parameter preflight without a control token

## Context

The source backend deliberately permits a small unauthenticated read-only
surface on loopback. The canonical WING read-only preflight runner uses
`query_mixer_parameter`, but that message was missing from the backend
read-only allowlist, so a physically connected observation-only session failed
before its first parameter read.

## Options considered

1. Add `query_mixer_parameter` to the existing read-only allowlist.
2. Give the preflight runner a live-control authentication token.
3. Bypass the backend and query WING directly.

## Decision

Allow `query_mixer_parameter` in unauthenticated loopback observation mode.
The handler continues to enforce its explicit address/value-type registry and
does not expose any write method.

## Why this won

It restores the documented backend-mediated preflight without creating a
control session, changing operator mode, or widening live-write authority.

## Rejected alternatives

An authentication token would grant a broader control-plane capability than a
read-only proof needs. Direct OSC would bypass the canonical backend boundary.

## Implementation plan

Add the message type to `UNAUTHENTICATED_READ_ONLY_MESSAGE_TYPES`, cover it in
the existing dispatch regression test, and pace large physical-readback plans
below the backend's ten-frame-per-second ingress limit.

## Test plan

Run the focused server test, the WING read-only runner tests, and a real
observation-only preflight with physical readback.

## Risks and rollback

The risk is limited to disclosure of allowlisted console parameter values to a
local process. Remove the allowlist entry to restore the previous fail-closed
behavior; no console state is changed by this path.
