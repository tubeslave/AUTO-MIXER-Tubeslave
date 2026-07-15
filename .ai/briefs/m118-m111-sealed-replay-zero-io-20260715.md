# M118 M111 Sealed Replay Zero-I/O Contract

## Task

Restore the M111 current-release compatibility replay to exact zero network and
audio I/O attempts before any attended WING write.

## Evidence

- The proof-owned replay test exits successfully, but the outer verifier sees
  `socket_inet_construct=1` and fails `replay_io_guard_counters_exact_zero`.
- The captured call stack is an import-time `requests -> urllib3` IPv6
  capability probe. `urllib3.util.connection._has_ipv6()` constructs an
  `AF_INET6` socket and catches the guard exception.
- Console writer counters remain zero; no backend transport, WING, listener or
  audio device was contacted.
- A replay artifact root below the group-writable checkout also makes the
  durable result store fail closed. A private `0700` runtime root works.

## Required outcome

1. Disable only the third-party import-time IPv6 capability probe while the
   socket guards are active, preload that dependency, and restore the real
   capability before any project import.
2. Keep every constructor/bind/connect/send/audio guard active.
3. Make the isolated pytest fail directly unless every I/O counter is zero.
4. Run the sealed replay from a private runtime artifact root.
5. Recreate the source-bound owner challenge after the committed change.
6. Do not contact WING, backend transport, CoreAudio or grant live authority.
