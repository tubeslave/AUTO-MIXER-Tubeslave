# dLive MIDI/TCP v2.0 golden vectors

Primary source: [Allen & Heath dLive MIDI Over TCP/IP Protocol, firmware
V2.0+](https://www.allen-heath.com/content/uploads/2024/06/dLive-MIDI-Over-TCP-Protocol-V2.0.pdf),
published June 2024. The downloaded review copy used on 2026-08-03 had SHA-256
`aca9e637a54912e560f14f18c42ad7d481f77b3172d3c2cabd684d36b309c253`.

All numbers below are hexadecimal unless stated otherwise.

## Endpoint contract

| Endpoint | Plain TCP | TLS | Identity meaning |
|---|---:|---:|---|
| MixRack | 51325 | 51327 | An open port is only a candidate, not model/firmware proof. |
| Surface | 51328 | 51329 | An open port is only a candidate, not model/firmware proof. |

TLS authentication begins with `UserProfile, UserPassword`, where
`UserProfile` is one byte `00..1F`; success is the six ASCII bytes `AuthOK`.
Do not store the password in config, logs, WAL, receipts, or reports.

## Channel and value contract

- Base MIDI channel `N` is `0..B` (MIDI channels 1..12). Channel types use
  offsets `N..N+4`; no wraparound is permitted.
- Input 5 is selector `04` on `N`.
- Faders are 7-bit, parameter ID `17`: `-inf=00`, `-20 dB=43`, `-5 dB=61`,
  `0 dB=6B`, `+5 dB=74`, `+10 dB=7F`.
- The protocol writes repeated status bytes as `[BN]` and describes their
  omission using MIDI running status. AUTO-MIXER emits the documented compact
  form and accepts both compact and full-status streams. This is a software
  conformance choice, not evidence that the connected console accepts writes.

## Byte vectors for input 5 on base channel 1 (`N=0`)

| Operation | Bytes |
|---|---|
| Set fader -5 dB | `B0 63 04 62 17 06 61` |
| Get fader | `F0 00 00 1A 50 10 01 00 00 05 0B 17 04 F7` |
| Mute on + release | `90 04 7F 90 04 00` |
| Mute off + release | `90 04 3F 90 04 00` |
| Get mute | `F0 00 00 1A 50 10 01 00 00 05 09 04 F7` |
| Get name | `F0 00 00 1A 50 10 01 00 00 01 04 F7` |

## Promoted boundary

The server connects read-only. Only typed input fader/mute methods have an
implemented encoding, and they still require an explicit attended-write
authority supplied outside the normal server/factory path. Name is read-only.
Scene, routing, preamp, phantom, pad, EQ/HPF, sends, pan, compressor, gate,
delay, polarity, name and colour writes fail closed with zero transport bytes.

These vectors prove software encoding only. They do not prove the connected
console, firmware, show MIDI base channel, patch, readback, rollback, or safe
acoustic state.
