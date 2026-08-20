# ADR: dLive MIDI/TCP readiness boundary

## 1. Context

The repository exposes `DLiveClient`, but the implementation used a 14-bit
NRPN model and several parameter IDs that do not match Allen & Heath
`dLive MIDI Over TCP/IP Protocol`, firmware v2.0 and later, published June
2024. The current WING milestone remains externally gated and must not be
reinterpreted as dLive evidence.

Official primary source:
https://www.allen-heath.com/content/uploads/2024/06/dLive-MIDI-Over-TCP-Protocol-V2.0.pdf

## 2. Options considered

1. Keep the current client and warn: unsafe because wrong bytes stay reachable.
2. Remove direct dLive support and use Mixing Station only: safe but leaves an
   already exposed backend path unresolved.
3. Narrow the direct client to an official, tested, deny-by-default core.

## 3. Decision

Choose option 3. Normal server/factory connections are read-only and have no
mutation-enabling constructor flag. The only first-write surface is a separate
restore-only input-fader proof controller with a process-private one-shot
capability. Mute encoding is not exposed as a write. Channel-name requests are
read-only. Every other family fails closed without sending until it has exact
official mapping plus attended apply/readback/rollback evidence.

The proof target must remain at or below the fresh preimage after conversion to
the exact 7-bit MIDI value. There is no tolerance that can authorize a small
increase or a cut beyond the operator's dB bound. Targets mapping to `-inf` and
preimages above `0 dB` are rejected before the apply WAL boundary, because the
former is not a finite bounded proof and the latter cannot be restored without
crossing the project-wide fader ceiling.

The proof controller shares the per-user `console_writer.lock` with WING,
writes a hash-chained/file-and-directory-fsynced dLive WAL before every send
boundary, performs fresh double readback and exact compare-and-swap rollback,
then revokes its capability and disconnects. A separate recovery command can
only restore the WAL preimage; it cannot replay the proof target.

Each proof approval is also bound to one exact owner-controlled GET-only
artifact by absolute path and SHA-256. The controller validates its endpoint,
port, MIDI base, user profile, input identity, muted state, fader preimage,
zero-write authority and clean teardown before route lookup or TCP, then
requires fresh physical readback to match the artifact's exact 7-bit preimage.
The operator confirmation itself includes a SHA-256 binding over the exact
artifact, endpoint, channel identity, console attestation, target/raw value and
maximum cut plus a unique approval ID. The former reusable generic literal
cannot authorize a proof. A durable `prepared` WAL row consumes that ID; it
cannot start a second transaction.

## 4. Why this won

It removes reachable speculative bytes while preserving a minimal path for the
first attended proof. It also lets configuration and parser behavior be tested
without a console.

## 5. Rejected alternatives

- No warning-only compatibility mode.
- No guessed NRPN for pan, compressor, gate, delay or polarity.
- No implicit input-channel to preamp-socket mapping.
- No scene, routing, phantom, pad, gain, PEQ/HPF or send live scope in this
  checkpoint.

## 6. Implementation plan

1. Pin v2.0 constants and golden vectors in project docs/tests.
2. Validate Surface/MixRack, TLS/port and MIDI base channel before sockets.
3. Implement 7-bit fader conversion, official NRPN order, channel selectors,
   mute and channel-name query/reply bytes.
4. Parse fragmented TCP MIDI with running status.
5. Require fresh post-request timestamps for readback.
6. Add a transport-free new-network preflight and attended runbook.
7. Keep unsupported methods present only as explicit zero-send blockers for
   interface compatibility.
8. Remove boolean write authority from `DLiveClient`; bind one apply and one
   exact rollback to a short-lived attended approval.
9. Add global writer exclusion, durable WAL, interrupted-run recovery and
   artifact receipts outside the repository.
10. Bind the write proof to an exact read-only artifact and close every
    pre-transport emergency interval with a terminal zero-send WAL record.
11. Preserve endpoint, MIDI base channel and user profile through saved config,
    direct connect, discovery/auto-connect, soundcheck, UI and product-state
    surfaces; never let an attended identity silently fall back to defaults.
12. Validate the exact artifact, bounded cut and apply/rollback packets with a
    zero-route/zero-TCP plan command before requesting the SHA-bound operator
    confirmation. This command grants no execution authority.

## 7. Test plan

- official byte vectors for fader, mute, get-fader and get-name;
- all channel-selector families as pure functions;
- fragmented/running-status receive sequences;
- stale-cache cannot satisfy readback;
- unsupported methods produce zero sends;
- normal fader/mute setters produce zero sends;
- wrong/expired approval, route, identity, mute state or writer lock blocks
  before mutation;
- WAL `prepared` and send-boundary records precede transport and form a valid
  hash chain;
- apply exceptions attempt exact rollback; third-party/CAS drift is never
  overwritten;
- sub-step increases, over-bound cuts, `-inf` targets and preimages above
  `0 dB` are all zero-mutation failures;
- interrupted transactions block new proofs until exact-preimage recovery;
- artifact tamper, endpoint/state mismatch, missing user profile, symlink or
  non-canonical path blocks before route lookup and client construction;
- emergency interruption after `prepared` or apply boundary is terminal with
  zero sends; interruption after transport restores the exact preimage;
- confirmation-token changes on artifact, endpoint, channel or target change;
  the old generic literal blocks before route lookup and proof-runner entry;
- exact production artifact/target plan validates without route lookup, socket
  creation, MIDI send or authority;
- invalid endpoint/TLS/base-channel inputs fail before socket creation;
- focused dLive suite, transport safety, WING rails, no-WING gate, full suite.

## 8. Risks and rollback

Software correctness does not prove the target console, firmware, show MIDI
base channel, patching, credentials or network. Roll back this change set if
golden-vector or regression tests fail. During hardware work, stop on identity,
timeout, mismatch, disconnect or stale readback; do not advance from read-only
to mutation. The implemented proof runner does not itself promote dLive live
control: only an accepted physical artifact and a separate owner decision may
change the capability matrix.

### 2026-08-04 physical NRPN correction

The first attended restore-only attempt against `CDM32+C3500` sent the
MIDI-valid full-status form `BB 63 00 BB 62 17 BB 06 65`. The console returned
the unchanged preimage four times; the terminal artifact is
`restored_after_unverified_apply`, with no unknown state and clean teardown.
The official dLive table denotes later status bytes as `[BN]`, so the exact
transmit vector is the running-status form `BB 63 00 62 17 06 65`. The encoder
and golden vector now use that seven-byte form; the parser continues accepting
both forms. No retry is authorized by this decision: a new attended approval
and unchanged physical containment are required.

The subsequent approved retry used the corrected running-status packet and was
also a terminal unchanged-preimage no-op. Therefore the status-byte form was a
real conformance correction but is not established as the physical root cause.
Before another proof, the operator must explicitly confirm `Receive MIDI` is
enabled in `Utility / Control / MIDI`. The next proof must target the separately
verified C3500 Surface endpoint `192.168.3.71:51328`; every approval remains
single-use.

The next proof must use a newly generated Surface read-only artifact because
the earlier artifact predates the mandatory `user_profile` field. Its exact
path and SHA are part of the one-shot approval. This is an additional
fail-closed precondition, not evidence that `Receive MIDI` is enabled and not
live authority.

### 2026-08-04 paired-online Surface no-op

Director subsequently machine-observed CDM32 and C3500 paired online. Fresh
Surface and MixRack GET-only probes agreed on `KICK`, raw fader `106` and Mute
ON. Under a new exact approval, with Director closed and the physical safety
conditions renewed, the Surface endpoint accepted the TCP connection and one
`BB 63 00 62 17 06 65` send attempt at the socket layer, but neither the four
fresh GET readbacks nor the operator's physical observation showed movement.

This closes the approval as a terminal negative experiment; it does not justify
another packet variant. The project must not describe running status as a
physically proven root-cause fix. Before another mutation, collect evidence for
effective MIDI security/reboot state and distinguish known-good-client behavior
from AUTO-MIXER wire delivery. Normal dLive operation and every unsupported
mutation family remain zero-send/read-only.
