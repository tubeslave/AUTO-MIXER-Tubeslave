# Codex critique of the independent M111 proposal

## 1. Strongest part of the other proposal

The independent review identifies the key semantic invariant: migration may
make historical behavior usable as a narrow M006 input, but it cannot turn that
behavior into native runtime-identity-v2 provenance. Requiring a fresh child
process materially strengthens the compatibility claim.

## 2. Possible mistakes or blind spots

A child process alone is not enough. The parent must challenge it, bind the
prestart source manifest before import/construction, verify exact loaded module
identities and ensure output/result stores are isolated from canonical runtime
state. Otherwise process separation is mostly cosmetic.

## 3. Where the solution is too complex, fragile, or costly

Requiring a networked backend process would add port/authentication and ingress
complexity unrelated to the compatibility fact being tested. The child should
load and exercise the canonical handler out of process without opening a
WebSocket listener, WING transport or audio device.

## 4. What should be kept

Keep separate native/migration flags, the exact golden historical hashes, the
out-of-process boundary, zero-writer counters and the serial-bound owner
attestation as a hard gate.

## 5. Revised recommendation

Launch a sealed no-network child harness from a clean checkout. Bind it with a
parent nonce/challenge and prestart source manifest, execute the canonical
handler through its registered dispatch, and capture a result artifact in an
isolated temporary store. Accept migration only through a new aggregate M006
input flag; never set native provenance ready.
