# Independent security review of the Codex proposal

## 1. Strongest part

The proposal correctly treats UDP source tuples and loopback ports as location
constraints, not authentication, and chooses fail-closed behavior rather than
inventing proof the protocols cannot provide.

## 2. Possible mistakes or blind spots

- WING authority had sibling bypasses in M110 exact-query, Main-mute, and
  USB-route readback helpers.
- Mixing Station also had direct client and auto-soundcheck adapter sinks.
- Expired WebSocket sessions could reach the emergency fast lane before the
  expiry check.
- JSONL line iteration needed bounded `readline(limit + 1)`, not a length check
  after an unbounded read.
- Replay batches needed validation before the first state transition.
- Generic transport ownership needed a strong object reference, not only an
  integer object id, and owner teardown needed to revoke the transport.

## 3. Where the proposal was too fragile

Quarantining only server entrypoints would leave direct transports callable.
Changing only the top-level WING verifier would leave other authority decisions
using raw mappings. Per-proposal replay rejection could allow a partial batch.

## 4. What should be kept

Keep token-required network startup, observational WING receipts, direct
Mixing Station quarantine, pre-parse resource bounds, literal audit evidence,
finite Safety Gate policy, loopback-only Virtual Mixer, and bounded GPT bridge.

## 5. Revised recommendation

Apply each control at the lowest shared sink, then retain server checks as
defense in depth. Add regression tests for the bypasses and preserve a
test-only trusted-verifier seam for transaction state-machine positive controls;
production OSC must never satisfy it.

## 6. Post-implementation review

The first patch review found two remaining blockers: mixer ownership checks
could race across an awaited disconnect broadcast, and configured dataset byte
caps stopped at path resolution instead of reaching every parse sink. The
revised implementation serializes connect/replace/disconnect under one async
lifecycle lock, passes the effective byte cap into all three model loaders,
checks the opened file descriptor, bounds classifier JSONL reads, and caps CSV
input lines, samples, and converted output. These controls require focused
regression tests before closure.

The final narrow review confirmed closure after transports began returning an
explicit verified teardown result, the server latched every non-`True` result,
regular files were opened with nonblocking/no-follow semantics, and NPZ loading
used one immutable bounded snapshot. No concrete bypass remained in the
reviewed remediation paths.
