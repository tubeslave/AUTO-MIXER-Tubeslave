# ADR: WING MVP Backend Port Local Socket Test Helper

## Context

The no-WING backend-port prestart gate must fail closed when an unexpected
listener is present before a WING session. Tests simulate this by binding a
local ephemeral socket and then running `check_port_free()`. Some environments
can deny local socket bind, which is not a failure of the no-WING gate itself.

## Options considered

1. Keep direct `socket.bind()` calls in every backend-port listener test.
2. Use a shared helper that skips only when local socket bind is unavailable.
3. Replace busy-port tests with mocks only.

## Decision

Use option 2.

## Why this won

It preserves real busy-port coverage on normal hosts while making the tests
honest about environments that cannot create the local test socket.

## Rejected alternatives

- Option 1 leaves environment-only failures in the no-WING backend-port tests.
- Option 3 removes useful real-socket coverage of the busy-port branch.

## Implementation plan

- Add `bind_local_test_server_or_skip()`.
- Replace backend-port listener tests' direct local socket binds with the
  helper.
- Add this report to the current WING blocker evidence and docs index.
- Validate focused tests and the full suite.

## Test plan

- Focused backend-port no-WING tests.
- Focused WING rails and Project Director tests.
- Project Director JSON check.
- Direction guard JSON check.
- `tools/memory/memory_lint.py`.
- `git diff --check`.
- no-WING quick check.
- Full pytest.

## Risks and rollback

Rollback restores direct socket binds in the affected tests and removes the
documentation references. This change does not contact WING, mutate console
state, grant live authority, or close the current blocker.
