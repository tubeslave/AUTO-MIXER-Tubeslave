# ADR: WING MVP Response Types Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`. The
preflight runner and evidence intake must agree on which backend response types
are valid for each read-only request. Local duplication can let runner behavior
and artifact validation drift.

## Options considered

1. Keep local expected response sets in runner and evidence.
2. Move expected response types into the shared preflight contract.
3. Remove response-type summary checks and rely only on request IDs.

## Decision

Use option 2.

## Why this won

Unexpected response handling is a fail-closed safety boundary. The same contract
should define response expectations for runner runtime checks and evidence
event-log validation.

## Rejected alternatives

- Option 1 permits drift between real-run behavior and artifact review.
- Option 3 keeps request identity checks but weakens response-shape review.

## Implementation plan

- Add response-type tuples for operator-mode, state, and readback responses to
  the shared preflight contract.
- Derive runner expected event sets from those tuples.
- Derive evidence expected event sets from those tuples.
- Add focused tests for shared response-type binding.

## Test plan

- Compile touched modules.
- Focused runner, evidence, and no-WING tests.
- no-WING quick on an unused backend port.
- Project Director and direction guard checks.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

The main risk is stricter validation exposing stale local response handling.
That is intended. Rollback would restore local sets, but would reopen drift
between real-run fail-closed behavior and evidence review.
