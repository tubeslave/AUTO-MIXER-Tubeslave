# Codex Proposal: M118 M111 Sealed Replay Zero-I/O

## 1. Short thesis

Prevent urllib3's import-only IPv6 probe without weakening the replay's absolute
socket/audio guards, and make zero attempted I/O an assertion inside the test.

## 2. What I understood

The application did not contact WING or a network endpoint. A dependency import
attempted to construct an IPv6 socket solely to detect local IPv6 support, the
guard raised, and urllib3 swallowed the exception. The test therefore reached
its safe terminal report while the outer verifier correctly rejected the
nonzero constructor counter.

## 3. Proposed solution

- Install every socket guard first, temporarily set `socket.has_ipv6=False`,
  preload `urllib3.util.connection`, and restore the original capability before
  importing any project module.
- Retain patched socket constructor, bind, connect, listen and send methods, so
  any application I/O attempt still fails and is counted.
- Assert all I/O counters are zero before writing the replay report.
- Use a private runtime artifact root rather than the group-writable checkout.

## 4. Likely files to touch

- `tests/test_m005_current_release_migration_replay.py`
- M118 council, ADR, plan and truth surfaces
- CP05 marker pair after the source checkpoint

## 5. Alternatives considered

- Pre-import urllib3 before installing guards: rejected because the IPv6 probe
  could perform unobserved socket/bind I/O.
- Ignore constructor-only attempts in the verifier: rejected because it weakens
  the exact-zero proof contract.
- Patch urllib3 or pin an older dependency: rejected as broader and brittle.

## 6. Risks

The scoped capability override must be restored even if the dependency import
fails, and must not disable guards. A tracked test change also invalidates the
existing source-bound challenge and CP05 marker.

## 7. Test plan

Run the isolated test, tool/verifier tests, a clean-source sealed replay with
all counters zero, WING rails and the standard full suite. Recreate the M111
challenge only after the final committed source marker verifies.

## 8. Where the other agent may disagree

A reviewer may prefer dependency injection in production imports. That is
larger than necessary because the behavior belongs exclusively to the sealed
proof harness and no production network path is being changed.
