# ADR: WING MVP Handoff Prestart Summary Placeholder

Date: 2026-06-13
Status: accepted

## Context

The read-only WING preflight runner requires a passed no-WING prestart
`summary.json` for both offline plan generation and real read-only capture. The
operator handoff packet must not lag behind that runner contract.

## Options considered

1. Document `--prestart-summary` only in the runbook.
2. Add `--prestart-summary` to handoff packet commands and no-WING validators.
3. Remove the runner requirement until hardware is available.

## Decision

Use option 2. The handoff packet declares
`backend_port_prestart_summary_required: true`, carries a prestart summary path
placeholder, and passes that path to both plan-only and capture commands.
no-WING release checks validate the path match.

## Why this won

The handoff packet is the operator-visible gate. If it omits a runner-required
argument, an operator can follow the packet and hit a fail-closed path that was
not visible in the packet validation.

## Rejected alternatives

Runbook-only documentation was too weak because generated artifacts would still
drift. Removing the runner requirement weakened the backend-port prestart
boundary.

## Implementation plan

Add a shared placeholder constant, update handoff command builders, strengthen
no-WING validators, write a dedicated prestart summary artifact for offline
release checks after the backend-port gate passes, and update tests/docs.

## Test plan

Focused tests must prove command binding and fail-closed validation. Wider
guard tests must keep Project Director on the first WING blocker.

## Risks and rollback

If the prestart path binding blocks a legitimate operator flow, revert the
handoff/no-WING validator changes and keep the runner requirement while the
operator packet is revised. Do not relax live-write or readiness boundaries.
