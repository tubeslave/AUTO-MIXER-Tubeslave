# Independent Proposal: M118 M111 Sealed Replay Zero-I/O

Independent read-only Codex reviewer fallback; this is not a Kimi opinion.

## 1. Short thesis

Preserve the real IPv6 runtime capability and either model the exact loopback
probe explicitly or scope any override exclusively to the proven third-party
import, restoring it before project code executes.

## 2. What I understood

The nonzero counter is not WING contact, but it is a real socket construction
and attempted loopback bind. Simply leaving `socket.has_ipv6=False` throughout
the replay would create an artificial runtime and could hide conditional IPv6
paths.

## 3. Proposed solution

- Preferred long-term: isolate the replay from pytest/plugin dependency imports.
- Bounded current fix: install all socket guards, suppress only urllib3's
  import-time capability probe, restore `socket.has_ipv6`, then import project
  modules with the original runtime capability.
- Keep every prohibited counter exactly zero and assert that inside the test.
- Keep the durable result store strict; use a private artifact root.

## 4. Likely files to touch

- `tests/test_m005_current_release_migration_replay.py`
- Council/ADR/plan and current truth records

## 5. Alternatives considered

- Permit one `::1:0` probe and record it separately.
- Globally disable IPv6 for the replay.
- Replace pytest with a standalone proof child.

## 6. Risks

A broad capability override weakens parity. Allowing one socket probe expands
the verifier and needs careful negative tests. A standalone child is stronger
but larger than the immediate blocker.

## 7. Test plan

Require original IPv6 capability restoration, exact-zero counters, isolated
pytest success, sealed clean-source replay and outer verification.

## 8. Where the other agent may disagree

Codex may consider a replay-wide IPv6 override harmless because all socket
constructors remain patched. The narrower preload-and-restore boundary is more
defensible and costs little.
