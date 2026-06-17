# ADR: WING read-only plan-output write errors deny readiness explicitly

## Context

When `--plan-only --plan-output` cannot write the requested expected-plan file,
the runner blocks before any backend/WING WebSocket. That blocked output already
denies execution and live authority.

## Options considered

1. Rely on the blocked status plus `live_authority_granted: false`.
2. Also emit `readiness_closure_allowed: false`.

## Decision

Use option 2. Plan-output write-error stops now explicitly deny readiness
closure.

## Why this won

The expected-plan artifact is part of the evidence chain. If it cannot be
written, readiness closure must be impossible and machine-readable without
inferring from `status`.

## Rejected alternatives

Inferring readiness denial from the blocked status was weaker than the rest of
the WING preflight surfaces, which prefer explicit authority and readiness
fields.

## Implementation plan

Add `readiness_closure_allowed: false` to the plan-output write-error block and
assert it in the regression.

## Test plan

Run the targeted plan-output write-error test, focused WING/no-WING suite,
no-WING quick, `git diff --check`, and the standard pytest gate.

## Risks and rollback

Risk is minimal because this only adds an explicit false field on an already
blocked path. Rollback is to remove the field and assertion.
