# ADR: M118 M111 Sealed Replay Zero-I/O

## 1. Context

The clean-source M111 replay reached its safe test terminal with zero console
writer calls and no backend/WING/audio contact, but the outer verifier rejected
`socket_inet_construct=1`. A captured stack proved that importing the
Mixing Station REST dependency loads urllib3, whose IPv6 capability detection
constructs and attempts to bind an `AF_INET6` loopback socket while swallowing
the guard exception. The main checkout is also group-writable, so the durable
result store correctly rejects replay roots below it.

Kimi CLI returned HTTP 402 in read-only plan mode. An independent read-only
Codex reviewer supplied the fallback proposal and critique.

## 2. Options considered

- Ignore or explicitly allow the one loopback capability probe.
- Disable IPv6 capability for the entire replay.
- Pre-import urllib3 before installing guards.
- Install guards, suppress only the dependency probe, restore capability before
  project imports, and retain exact-zero verification.
- Replace pytest with a standalone stdlib replay child.

## 3. Decision

Install all socket guards first. Temporarily expose `socket.has_ipv6=False` only
while preloading `urllib3.util.connection`, restore the original value in
`finally`, and only then import project modules. Assert every I/O counter is
zero inside the test and retain the independent outer exact-zero verifier. Run
sealed artifacts only below a private non-group-writable runtime root.

## 4. Why this won

It prevents the known third-party import probe without granting any socket
exception or changing the runtime capability observed by project code. It is
smaller than a standalone replay rewrite and preserves the existing safety
schema.

## 5. Rejected alternatives

Allowing the loopback probe weakens exact-zero semantics. A replay-wide override
reduces parity. Pre-importing before guards permits unobserved I/O. A standalone
child remains a valid later hardening option but is not required to close the
reproduced blocker.

## 6. Implementation plan

Patch the sealed replay test, add the direct zero assertion, run targeted tests,
commit and rebind CP05, then generate a new clean-source M111 challenge and run
the sealed replay under `~/.automixer/runtime`.

## 7. Test plan

Run the isolated proof test, migration tool tests, clean-source sealed replay,
outer migration verification, WING MVP rails and the standard full suite.

## 8. Risks and rollback

If the scoped preload is not restored, project parity is weakened; the explicit
restoration assertion prevents that. If a dependency changes its probe, any
guard counter fails the test. Revert M118 and its paired CP05 rebind together;
never reuse a challenge bound to the reverted source.
