# ADR: WING MVP Request Response Map Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`. The
preflight runner and evidence intake must agree on which response types are
valid for each read-only request type. The response type tuples were shared, but
the request-to-response mapping was still local to the code paths.

## Options considered

1. Keep local request-to-response mappings.
2. Move the request-to-response mapping into the shared preflight contract.
3. Validate only the raw response type allow-list, independent of request type.

## Decision

Use option 2.

## Why this won

The mapping is the exact fail-closed contract for `unexpected_response_type`.
It must be identical between runtime handling and captured artifact validation.

## Rejected alternatives

- Option 1 permits drift between runner behavior and evidence review.
- Option 3 would accept response types in the wrong request context.

## Implementation plan

- Add a shared mapping from request type to expected response type tuple.
- Derive runner `EXPECTED_*` sets from that mapping.
- Derive evidence response checks from that mapping.
- Add focused tests for shared mapping binding.

## Test plan

- Compile touched modules.
- Focused runner, evidence, and no-WING tests.
- no-WING quick on an unused backend port.
- Project Director and direction guard checks.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

The main risk is stricter validation surfacing stale local response handling.
That is intended. Rollback would restore local mappings, but would reopen drift
between real-run fail-closed behavior and evidence review.
