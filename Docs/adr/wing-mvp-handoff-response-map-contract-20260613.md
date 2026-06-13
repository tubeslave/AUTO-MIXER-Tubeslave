# ADR: WING MVP Handoff Response Map Contract

## Context

The WING read-only preflight handoff packet guides the operator through the
hardware step. The expected plan now exposes the accepted request-to-response
mapping, but the handoff packet did not expose or validate the same mapping.

## Options Considered

1. Leave the response mapping only in the expected plan.
2. Add the response mapping to the handoff packet and validate packet/plan
   agreement.
3. Replace the handoff packet and expected plan with a larger shared schema.

## Decision

Add `expected_response_types_by_request_type` to the handoff packet, render it
in markdown, and fail no-WING validation when the packet differs from the shared
contract or the expected plan artifact.

## Why This Won

It improves operator review of the real WING step and catches stale handoff
packets without changing any live command path or granting authority.

## Rejected Alternatives

Plan-only mapping was rejected because the handoff packet is the operator's
primary hardware-step artifact. A broader schema migration was rejected as too
large for this focused readiness hardening step.

## Implementation Plan

Derive the handoff map from
`WING_READONLY_PREFLIGHT_EXPECTED_RESPONSE_TYPES_BY_REQUEST_TYPE`, render it,
validate it in `tools/no_wing_release_check.py`, and cover happy-path plus drift
tests.

## Test Plan

Run compile checks, focused handoff/no-WING tests, no-WING quick, project
direction guards, memory lint, diff check, and the standard pytest suite.

## Risks and Rollback

Risk is limited to stale handoff packets failing closed. Roll back by reverting
this patch; no live-write path is introduced.
