# ADR: M111 truthful M005 current-release migration

## 1. Context

The owner states that M005 behavior is closed and authorizes narrow M006. The
accepted physical artifact from 11 July proves baseline readback, physical
cable loss, stale readback unavailability, a blocked apply and unchanged
reconnect readback. Later runtime-identity-v2 hardening added prestart source,
proof-owned process, initial connection and console identity requirements that
cannot be reconstructed after the run.

The current packet verifier therefore reports 43 failures. Retrospectively
adding the missing fields would fabricate provenance. Repeating the cable pull
would create native evidence, but the owner did not request another M005 run.

## 2. Options considered

1. Rewrite or annotate the historical summary as current provenance.
2. Treat the owner's closure statement as a provenance waiver.
3. Require a new physical cable-pull run.
4. Add a separate migration route combining immutable historical hardware
   facts, proof-owned current-code compatibility evidence and fresh console
   identity attestation.

## 3. Decision

Choose option 4, with option 3 remaining the preferred native-proof fallback.

Migration uses `automixer.m005_current_release_migration.v1` and is a separate
acceptance route. It must not set native `m005_provenance_ready` or native
`m005_current_release_accepted` to true. M006 uses a new aggregate
`m005_m006_input_accepted` flag and records the selected route.

The route requires all of the following:

- exact owner-accepted historical artifact paths and immutable SHA-256 values:
  summary `a5830f94cf35dad9af96127837efe04fc345ff65c7c319dc3aa216b57a047e64`
  and events `8d40010f529f553537eee821f04e1fab585086ff4d28d8763bd4256bb2ffeb45`;
- the exact accepted M006 owner-decision bytes, SHA-256
  `2d17ad2181fd7ea4ed7d730d8bdae30b155382559cf9b62c45440953a63bc018`,
  its packet/proof/reviewer bindings, and the historical preflight source
  `artifacts/wing_readonly_preflight/20260702T095758Z_wing_readonly_preflight/summary.json`;
- a schema-specific ordered extraction of baseline, operator cable-loss
  confirmation, unavailable stale readback, safely blocked apply, reconnect
  physical readback, target equality and absence of forbidden/rollback writes;
- a clean-current-source prestart manifest with HEAD/tree/runtime components
  and identities for the replay harness, backend and M006 verifiers;
- a proof-owned out-of-process no-network child, challenged by the parent and
  bound before server construction to the prestart manifest, PID/start/nonce
  and exact critical loaded-module `__file__` identities;
- canonical current `create_operator_proposal` and
  `apply_operator_proposal` replay for bounded negative fader and gain inputs
  under both stale-cache and disconnected global-hazard conditions, with
  blocked terminals, result-digest binding where evidence is persisted,
  `no_console_send_proven=true` and zero calls to every console
  writer/setter/raw-send path;
- parent-observed invocation metadata, exact interpreter/command identity,
  isolated runtime/result/manual-write roots, confined regular artifact files,
  and active socket/audio guards whose call counters must remain exactly zero;
- an identity-only owner attestation generated only after a fresh 24-hour
  challenge and bound to its unique migration run id, nonce, raw challenge
  hash, current HEAD/tree/source binding, actual WING Rack serial, firmware
  3.0.5, `same_console_as_historical_m005=true`, reviewer and bounded review
  time, explicitly stating that it is not a historical runtime query; and
- false live-authority, live-write, backend/WING contact, readiness-closure and
  readiness-matrix mutation flags on every generated/verifier artifact.

The replay child may use an isolated test harness to construct the canonical
server and result/WAL stores, but it must not open a listener, WING/OSC socket,
audio device or canonical per-user console runtime store.

The replay process installs AF_INET/AF_INET6 socket and sounddevice stream
constructor guards before importing any project/backend module. The exact
module identity set includes the migration audit helper itself.

The packet verifier prefers native current M005 when both routes are valid. If
an explicit migration path is supplied and that migration is invalid, the
packet fails closed even if native evidence would otherwise pass. A migration
decision must explicitly bind both `m005_acceptance_route=verified_migration`
and the exact `m005_migration_summary` path. The migration packet and decision
must also retain the exact canonical historical M005 summary/events paths;
repo-relative substitutes are rejected. Legacy/native decisions may omit the
two migration fields for backward compatibility.

## 4. Why this won

It preserves the physical facts the owner already accepted without pretending
that modern runtime metadata existed in the past. A fresh process proves which
current code was loaded, while the historical artifact remains the sole source
of physical cable-loss facts. The owner challenge binds the claimed console
identity without mislabelling an attestation as a hardware query.

## 5. Rejected alternatives

- Historical JSON mutation violates immutable audit evidence.
- Owner intent alone cannot prove current safety-code compatibility.
- In-process replay is too easy to self-affirm around the real handler boundary.
- Networked replay is unnecessary and adds a transport surface; the required
  current-code fact can be proven by a sealed no-network child.
- Migration cannot authorize EQ, dynamics, gate, pan, FX, routing, mute,
  master, scenes, gain-up, fader boosts or full auto.

## 6. Implementation plan

1. Add a narrow migration generator/verifier and sealed child replay harness.
2. Add exact historical parser/hash and owner-challenge validation.
3. Add optional migration inputs to M006 packet and decision verifiers while
   preserving the native route, native diagnostics, and explicit route/path
   binding in the owner decision.
4. Emit a blocked migration template until a fresh source/run-bound challenge
   and the real serial-bound owner attestation exist.
5. Run focused mutations, full validation and no-WING checks from a clean
   checkout, then perform one finite CP05 marker rebind after all tracked edits.

## 7. Test plan

- Flip every byte/path/hash/order/target/status in historical evidence.
- Mutate source HEAD/tree/file identities and child PID/nonce/challenge/module
  identities; reject in-process or unbound replay output.
- Force stale-cache and disconnected scenarios for both accepted kinds and
  assert exact bounded negative requests, blocked terminals, result bindings,
  no-send evidence, and zero writer/setter/raw-send/socket/audio calls.
- Mutate every console identity and reviewer field.
- Reject stale/replayed challenge, source/run mismatch, unsafe run ids,
  path escape and symlink substitution.
- Exercise full build -> confined path verifier round-trip, then mutate replay,
  invocation and owner snapshots.
- Verify native proof still works unchanged; migration is selected only when
  fully ready; absence of attestation remains blocked.
- Verify M006 still accepts exactly fader and gain-down readiness rows and all
  transport/authority flags remain false.
- Run focused tests, WING rails, full backend suite, frontend tests/build,
  five-store no-WING gate and independent diff review.

## 8. Risks and rollback

The main risk is turning migration into a silent provenance waiver. Route-
specific status fields, exact golden hashes, a proof-owned child and a hard
owner-attestation gate mitigate it. Rollback removes the migration route while
leaving native current M005 verification unchanged. Historical evidence is
never rewritten in either direction.
