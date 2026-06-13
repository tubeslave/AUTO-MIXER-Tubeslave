# ADR: WING MVP Handoff Allowed Request Contract

## Context

The WING read-only preflight plan exposes the allowed WebSocket request types
and the request-id policy that protects physical readback identity. The
operator handoff packet did not expose or validate those same fields.

## Options Considered

1. Leave allowed request types and request-id policy only in the expected plan.
2. Add both fields to the handoff packet and validate packet/plan agreement.
3. Replace the handoff packet and expected plan with a larger shared schema.

## Decision

Add `allowed_request_types` and `request_id_policy` to the handoff packet,
render them in markdown, and fail no-WING validation when either field differs
from the shared contract or expected plan artifact.

## Why This Won

It improves operator review of the real WING step and catches stale handoff
packets without changing live command paths or granting authority.

## Rejected Alternatives

Plan-only fields were rejected because the handoff packet is the operator's
primary hardware-step artifact. A broader schema migration was rejected as too
large for this narrow readiness hardening step.

## Implementation Plan

Derive the handoff fields from
`WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES` and
`WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY`, render them, validate them in
`tools/no_wing_release_check.py`, and cover happy-path plus drift tests.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, project
direction guards, memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to stale handoff packets failing closed. Roll back by reverting
this patch; no live-write path is introduced.
