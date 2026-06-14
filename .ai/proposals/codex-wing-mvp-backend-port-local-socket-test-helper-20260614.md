# Codex Proposal: WING MVP Backend Port Local Socket Test Helper

## Short thesis

Keep no-WING backend-port listener tests focused on prestart gate semantics by
skipping only the local socket setup when the environment forbids it.

## What I understood

The active WING blocker is still the read-only connection proof. The no-WING
backend-port gate remains a key safety precondition. The tests should remain
strict about listener confirmation and fail-closed behavior, but they should not
fail because a host environment disallows binding a local ephemeral test port.

## Proposed solution

Use `bind_local_test_server_or_skip()` for backend-port listener tests that need
an occupied port. The helper binds `127.0.0.1:0`, closes the socket and skips
the test only on `PermissionError`, then returns the socket for normal
`listen()` and `check_port_free()` assertions.

## Likely files to touch

- `tests/test_no_wing_release_check.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/INDEX.md`
- Council trail files for this task.

## Alternatives considered

- Leave direct `socket.bind()` calls in each test.
- Mock the whole port-listener path instead of binding a real local socket.

## Risks

A skip can hide coverage in a severely restricted environment. The mitigation is
that skipping happens only when local socket bind raises `PermissionError`; on
normal hosts the same fail-closed assertions still run.

## Test plan

Run focused backend-port tests, focused Project Director/WING rail tests,
Project Director JSON, direction guard JSON, memory lint, diff check, no-WING
quick check, and full pytest.

## Where the other agent may disagree

Kimi may prefer pure mocking. I prefer keeping the local socket simulation where
available because it exercises the real busy-port branch while avoiding
environment-only failures.
