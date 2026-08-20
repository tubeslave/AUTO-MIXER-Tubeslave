# Codex proposal — DLIVE-READINESS-20260803

## 1. Short thesis

Replace the speculative dLive transport with a small protocol-v2 core, add a
transport-free new-network preflight, and keep every unverified family blocked
until attended physical evidence exists.

## 2. What I understood

The immediate need is safe connection preparation, not autonomous dLive live
mixing. The code must be truthful about what the public MIDI/TCP protocol can
address and what has actually been proved on the target system.

## 3. Proposed solution

1. Correct ports, channel selectors, 7-bit value scales, NRPN order, SysEx IDs,
   scene numbering and running-status parsing from the official v2.0 document.
2. Add explicit Surface/MixRack endpoint validation and restrict base MIDI
   channel to the documented `0..11` range.
3. Add active read-only GET requests with fresh timestamp matching for input
   fader, mute and name.
4. Disable protocol-unverified pan, dynamics, gate, delay and polarity sends.
5. Require an explicit input-to-preamp-socket mapping before gain/pad/48V can
   be addressed; keep 48V forbidden in the normal integration.
6. Add a pure no-socket preflight report for IP/port/TLS/base-channel/network
   identity and an attended read-only runbook.
7. Update architecture/tools/capability documentation and add regression tests
   based on official byte examples.

## 4. Likely files to touch

- `backend/dlive_client.py`
- `backend/mixer_client_base.py`
- `backend/handlers/connection_handlers.py`
- `backend/server.py`
- `backend/mixer_discovery.py`
- `config/automixer.yaml`
- `config/mixing_station/dlive_capabilities.yaml`
- `Docs/ARCHITECTURE.md`, `Docs/TOOLS.md`, a dLive runbook and ADR
- focused dLive/network tests

## 5. Alternatives considered

- Keep the client and only add warnings: rejected because wrong bytes remain
  reachable.
- Route all control through Mixing Station: useful later, but it does not fix
  the direct client already exposed by the backend.
- Implement all mixer processing through undocumented guesses: rejected; the
  public protocol does not establish those mappings.

## 6. Risks

- Existing tests encode the wrong protocol and must be replaced rather than
  preserved as evidence.
- Input channels are not physical preamp sockets; implicit numeric mapping can
  change the wrong head amp.
- A TCP-open result identifies a service, not the expected console/firmware.
- TLS may require a trusted CA/hostname and credentials that must not be stored
  in the repository.

## 7. Test plan

- official byte-vector unit tests;
- fragmented/running-status parser tests;
- stale-cache/fresh-readback tests;
- unsupported-family zero-send tests;
- endpoint/TLS/base-channel preflight tests;
- focused dLive suite, safety invariants, WING rails, no-WING quick gate and
  full repository suite after final truth writes.

## 8. Where the other agent may disagree

An alternate reviewer may prefer removing the direct dLive client entirely or
may argue for a larger generic MIDI codec. The proposed bounded rewrite keeps
the public integration working while reducing live scope to documented bytes.
