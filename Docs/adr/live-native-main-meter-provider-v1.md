# Live native Main meter provider v1

Date: 2026-09-24
Status: software-complete, HIL-required

## Context

The Main-tap PATCH_VERIFY gate already proves two things in software:

1. reserved USB return slots fresh-read as the expected `MAIN N` routes;
2. the returned post-console Main tap is level-coherent with an independent physical Main-meter observation.

The missing link was a concrete source for that independent physical meter observation. Repository search did not reveal a validated OSC leaf for Main meter level, and inventing one would violate the fail-closed migration contract.

## Protocol finding

Behringer / Music Tribe WING Remote Protocols 3.1 documents realtime meters on the native metering channel, not as ordinary OSC parameter leaves:

- native control connection: TCP port 2222;
- meter request channel selection: `DF D3`;
- client declares a UDP return port with token `D3`;
- a 4-byte caller report id is declared with token `D4`;
- meter collection starts with `DC` and ends with `DE`;
- Main collection token is `A3`, with zero-based wire index (`00` means MAIN 1);
- the console sends meter frames to the declared UDP port for about five seconds, approximately every 50 ms;
- each returned frame starts with the caller report id, followed by signed big-endian 16-bit meter words;
- level words are scaled at 1/256 dB.

The independent open-source `libwing` implementation corroborates the same request path: it opens the native TCP connection on 2222, declares an ephemeral UDP meter port, selects the native meter request channel, requests `Meter::Main`, and decodes returned signed big-endian 16-bit words.

## Decision

Add `backend/live_runtime/wing_main_meter.py` as a narrow read-only PATCH_VERIFY transport primitive.

`WingNativeMainMeterProvider`:

- opens a short-lived native TCP connection to WING port 2222;
- binds an ephemeral local UDP receive socket;
- requests exactly one Main meter collection;
- accepts only a datagram with the requested 4-byte report id and exact one-Main payload shape;
- decodes output L/R meter words in 1/256 dB;
- returns the louder output meter as `PhysicalMainMeterEvidence.peak_dbfs` with a monotonic receive timestamp;
- provides no RMS because the native legacy Main collection does not expose an independent RMS word;
- closes both sockets after the one-shot PATCH_VERIFY read;
- never changes WING parameters, routing, faders, EQ, dynamics, or mode.

The provider deliberately does not become a second persistent control connection. Continuous metering can be designed later if required by the show loop; this primitive exists only to close the startup proof gap.

## Safety status

This is software evidence, not yet physical HIL evidence.

Autonomous Main mutation remains blocked until a real WING test proves:

1. the native request/UDP return path works on the project's WING firmware;
2. MAIN 1 output L/R words correspond to the intended physical Main meter point;
3. the native meter detector/ballistics are sufficiently comparable to the sample-domain post-console USB tap for the configured coherence corridor;
4. timestamp skew under real network/audio load stays inside the PATCH_VERIFY window;
5. failure/timeout remains fail-closed.

The current coherence code calls this field `peak_dbfs`, but the exact detector/ballistics semantics of the WING meter require HIL calibration before the value can authorize Main actions. If HIL shows a systematic detector offset or different ballistics, the contract/policy must be corrected rather than widening tolerances blindly.

## Migration classification

### KEEP_CORE

- documented WING native transport semantics;
- signed big-endian meter decoding;
- read-only transport and timing primitives.

### ADAPT

- `backend/live_runtime/wing_main_meter.py` is the canonical read-only Main-meter adapter for PATCH_VERIFY until shared transport extraction is justified;
- `PhysicalMainMeterEvidence` remains the typed proof boundary.

### ARCHIVE

No legacy decision policy is reused. Any old code that treats cached/heuristic Main level as proof remains an ARCHIVE candidate after runtime references are severed.

### DELETE_AFTER_PROOF

None in this pass. Hardware proof is still outstanding.

## Test evidence

Focused tests cover:

- exact documented native request bytes for MAIN 1;
- escaping literal native `0xDF` data bytes;
- exact 20-byte one-Main response shape;
- signed big-endian 1/256 dB decode;
- report-id mismatch rejection;
- one-shot TCP + UDP provider lifecycle;
- timeout fail-closed behavior and socket cleanup.
