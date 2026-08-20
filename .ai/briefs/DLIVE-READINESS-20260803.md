# DLIVE-READINESS-20260803

## Goal

Prepare AUTO-MIXER for an operator-attended connection to an Allen & Heath
dLive system on a new network, beginning with zero-MIDI network checks and a
read-only console session.

## Requested outcomes

- review current dLive protocol, network, configuration, readback and rollback
  readiness;
- correct protocol and implementation bugs;
- close no-console validation and operator handoff gaps;
- document the exact attended path and remaining physical gates;
- preserve the current WING milestone and deny dLive live authority until
  fresh physical readback and rollback evidence exists.

## Evidence found before implementation

- `backend/dlive_client.py` claims 14-bit NRPN faders and parameter IDs that do
  not match the official dLive MIDI over TCP/IP v2.0 protocol.
- The official protocol uses 7-bit fader and PEQ/HPF values, NRPN parameter ID
  `0x17` for faders, and distinct Surface/MixRack rendezvous ports.
- Current channel-selection offsets omit the required note-number ranges for
  stereo buses, FX returns, mains, DCA and mute groups.
- Several client methods (pan, compressor, gate, delay, polarity) are not
  documented in the official MIDI/TCP v2.0 controllable-function list.
- The receiver does not implement MIDI running status even though dLive uses
  it.
- `set_fader_verified()` can accept stale cached state.
- the TLS fields accepted by `server.connect_dlive()` are dropped by the
  WebSocket handler, and endpoint type/port consistency is not validated.
- Existing dLive artifact contracts are shadow-only and do not provide a
  source-owned read-only capture runner or a new-network preflight.

## Safety boundary

- TCP port handshakes are permitted after explicit operator attendance;
- no MIDI mutation messages and no hardware discovery scan;
- unsupported or unverified parameter families fail closed;
- no dLive live authority, readiness closure or WING milestone changes;
- real-console work must be attended, source-bound, read-only first, and then
  use a separately approved one-parameter apply/readback/rollback proof.

After the operator reported subnet 3, a bounded TCP-only check found that the
Mac wired interface was still `192.168.1.101/24`; subnet 3 routed through a VPN.
No dLive port was contacted and no MIDI bytes were sent. Administrator action
is required to place the wired interface in `192.168.3.0/24`.

## Acceptance

- protocol builders and parser match official v2.0 examples;
- endpoint and network inputs fail closed before socket creation;
- fresh-query readback cannot be satisfied by stale cache;
- unsupported mutation families return `False` without sending bytes;
- focused dLive tests and repository safety rails pass;
- runbook and ADR name the remaining hardware evidence explicitly.
