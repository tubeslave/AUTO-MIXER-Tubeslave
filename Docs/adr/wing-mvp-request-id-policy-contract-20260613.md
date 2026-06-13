# ADR: WING MVP Request ID Policy Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Read-only physical readbacks must be tied to their originating
`query_mixer_parameter` requests. The policy for that requirement was duplicated
as a string in runner summaries, evidence validation, no-WING validation, and
test fixtures.

## Options considered

1. Keep local literals in each surface.
2. Move the request-id policy string into the shared preflight contract.
3. Remove the summary policy field and rely only on event-level request IDs.

## Decision

Use option 2.

## Why this won

The request-id policy is part of the real WING proof boundary. It should be
reviewed and changed through the same shared contract that defines proof schema,
runner identity, readback value types, and prestart safety rails.

## Rejected alternatives

- Option 1 permits silent drift between runner output and evidence/no-WING
  validation.
- Option 3 hides an important operator-review invariant from summary artifacts.

## Implementation plan

- Add `WING_READONLY_PREFLIGHT_REQUEST_ID_POLICY` to the preflight contract.
- Derive runner and evidence module aliases from that constant.
- Use the constant in offline plan and real-run summary output.
- Use the constant in no-WING prestart validation.
- Update focused tests to assert the contract binding and remove duplicated
  literals from fixtures.

## Test plan

- Compile touched modules.
- Focused runner, evidence, and no-WING tests.
- no-WING quick on an unused backend port.
- Project Director and direction guard checks.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

The main risk is stricter validation rejecting a stale expected plan. That is
intended because stale identity-correlation policy should block the real WING
preflight before the backend is contacted. Rollback would restore local
literals, but would reopen proof-contract drift.
