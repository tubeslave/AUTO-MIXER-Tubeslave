# ADR: WING MVP Allowed Request Types Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`. The
read-only preflight runner is allowed to send only operator-mode, state, and
explicit mixer-parameter readback requests. That allowed request list was
duplicated in multiple runtime and test surfaces.

## Options considered

1. Keep local allowed-request lists in each surface.
2. Move the allowed request types into the shared preflight contract.
3. Remove the `allowed_request_types` summary field and rely only on event logs.

## Decision

Use option 2.

## Why this won

The allowed request types define the proof surface for the real WING read-only
run. A future expansion should require an explicit contract change and should
be visible to runner, evidence, no-WING, and tests at the same time.

## Rejected alternatives

- Option 1 permits silent drift between runner output and validators.
- Option 3 hides an operator-review invariant that should remain visible in
  expected plans and summaries.

## Implementation plan

- Add `WING_READONLY_PREFLIGHT_ALLOWED_REQUEST_TYPES` to the shared preflight
  contract.
- Derive runner plan `allowed_request_types` from the constant.
- Derive evidence allow-lists from the constant.
- Validate no-WING generated plans against the constant.
- Update focused tests to assert the contract binding and remove fixture
  duplication.

## Test plan

- Compile touched modules.
- Focused runner, evidence, and no-WING tests.
- no-WING quick on an unused backend port.
- Project Director and direction guard checks.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

The main risk is stricter validation rejecting stale expected plans. That is the
correct failure mode because stale request-surface policy should block the real
preflight before the backend/WING session is used. Rollback would restore local
lists, but would reopen proof-surface drift.
