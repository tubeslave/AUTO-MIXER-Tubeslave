# Brief: WING MVP Backend Port Local Socket Test Helper

## Task

Make no-WING backend-port listener tests tolerate environments where binding a
local test socket is unavailable.

## Context

The active WING MVP blocker requires the no-WING prestart backend-port gate to
fail closed before an intended backend/WING session. Several tests simulate a
busy backend port by binding `127.0.0.1` on an ephemeral port. In restricted
test environments, local socket bind can raise `PermissionError`, which turns a
test-environment limitation into an unrelated failure.

## Scope

- Use one helper for local test socket creation in backend-port listener tests.
- Skip only when local socket bind is unavailable.
- Keep no-WING backend-port fail-closed semantics unchanged.
- Record the decision and validation.

## Non-goals

- No production no-WING checker behavior changes.
- No backend/WING contact.
- No live writes or live authority.
- No readiness closure.

## Test plan

- Focused backend-port no-WING tests.
- Focused WING rails and Project Director tests.
- Project Director and direction guard JSON checks.
- Memory lint, diff whitespace check, no-WING quick check, and full pytest.
