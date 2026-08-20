# dLive readiness review — 2026-08-03

## Goal

Connect AUTO-MIXER to Allen & Heath dLive as a second console and operate it
under the same safety principles used for WING Rack: explicit identity and
network binding, typed protocol messages, one writer, operator approval, WAL,
fresh physical readback, bounded change, rollback, emergency stop and attended
hardware proof. No old WING evidence or authority transfers to dLive.

## Outcome of this checkpoint

The previous dLive transport was not safe to use. It encoded a non-existent
14-bit fader protocol, used wrong parameter IDs, omitted MixRack ports and MIDI
running status, accepted stale cache as write confirmation, exposed raw MIDI
and speculative DSP writes, and did not carry endpoint/TLS settings through
all server/UI paths.

The direct client is now a narrow official-v2.0 implementation. Normal server,
factory, soundcheck and UI connections are read-only and cannot write. A real
attended read-only probe exists for one input's name/fader/mute, with two fresh
responses per field and a zero-authority artifact. A separate restore-only
fader proof control plane now owns one-shot approval, the same per-user writer
lock used by WING, a hash-chained/fsynced WAL, fresh readback, bounded cut,
compare-and-swap rollback, recovery and teardown. The network and bounded
read-only transport have now run against the connected MixRack. The canonical
runner observed stable readback bound to the operator-attested `CDM32+C3500`,
firmware `2.0` and Show `lv 1 master`; the original MixRack artifact was
physically accepted, and the Surface endpoint is separately GET-verified.

## Evidence layers

| Layer | Score | Current evidence | Missing gate |
|---|---:|---|---|
| Official protocol + offline codec | 92/100 | Official v2.0 vectors, 7-bit NRPN, selectors, running-status parser, targeted tests | Independent physical byte/readback comparison |
| Safe connection/control plane | 94/100 | Endpoint/port validation, read-only default, raw-send blocked, one-shot restore-only capability, shared writer lock and durable WAL | Attended hardware execution and independent artifact review |
| New-network reachability | 90/100 | `en7=192.168.3.59/24`; route to `192.168.3.70` uses `en7`; TCP `51325` open | Preserve exact interface binding during canonical run |
| Console identity | 75/100 | Operator attested `CDM32+C3500`, firmware `2.0`, Show `lv 1 master`; endpoint/base/name respond correctly | Physical Surface comparison; identity fields remain `machine_verified=false` |
| Physical readback | 90/100 | Canonical artifact has two fresh stable GET replies per field, zero writes and clean disconnect; operator confirmed the C3500 matches `KICK`, `-0.5827 dB`, mute off | Independent artifact review and a separate restore-only physical proof |
| Mutation + rollback | 30/100 | Multiple bounded attempts, including a paired-online Surface attempt, closed with unchanged exact preimage, terminal WAL and clean teardown | Effective inbound MIDI/security proof, known-good-client or wire capture, then a new approved apply/readback/exact rollback |

Overall dLive operational readiness at this checkpoint: **78/100**. The
network, operator attestation, canonical protocol readback and physical C3500
comparison are present, and the read-only artifact is accepted. A physical
mutation/rollback proof is still absent. It is **not** ready for live correction
or autonomous dLive control.

## Requirement-to-evidence matrix

| Requirement | Implemented boundary | Evidence | State |
|---|---|---|---|
| Exact console/network identity | Endpoint, IP, port, TLS, interface, MIDI base, user profile, model/firmware/show and input identity are bound into artifacts | Attended MixRack and Surface GETs; route/interface receipts | Partial: model/firmware/show are operator-attested |
| Official MIDI/TCP encoding | Official 7-bit fader codec, selectors and running-status parsing/transmit vector | Golden vectors and focused protocol tests | Software-verified; physical apply still unverified |
| Read-only normal operation | Generic send and all normal mutation families reject with zero bytes | Transport-safety/static tests; normal UI/server constructors expose no write flag | Verified offline |
| One writer | dLive proof and WING share `console_writer.lock` | Lock contention and recovery tests | Verified offline |
| Explicit approval | Literal, expiring, single-use approval whose token is SHA-bound to the exact artifact, endpoint, channel and target | Negative approval/tamper/expiry tests | Verified offline; no standing approval |
| Durable WAL | Hash-chained, file+directory-fsynced boundaries before transport | Fault-injection tests at prepared/apply/rollback boundaries | Verified offline |
| Fresh readback | Double GET, timestamp freshness and exact 7-bit comparison | Stale-cache and physical read-only artifacts | Read-only verified physically |
| Bounded mutation | One muted input, downward target only, maximum 6 dB cut, no `-inf`, no above-0 preimage | Boundary/quantization tests | Verified offline |
| Exact rollback/recovery | Compare-and-swap rollback only; recovery never replays target | Exception/interrupt/CAS/recovery tests | Verified offline; physical proof missing |
| Emergency stop/teardown | Zero-send terminalization before send; exact rollback after send; revoke and disconnect on every exit | Fault-injection tests and two terminal no-op attempts | Verified offline; successful physical apply absent |
| Operator-visible configuration | UI, persisted config, direct connect, auto-connect, soundcheck and state snapshot preserve endpoint/base/profile | Backend and frontend contract tests | Verified offline |
| Autonomous dLive correction | No capability exists | `live_authority_granted=false`, unsupported mutation families zero-send | Intentionally blocked |

## Corrections made

- Surface ports `51328/51329` and MixRack ports `51325/51327` are distinct and
  validated before socket creation.
- Base MIDI channel is limited to `0..11`; `N+4` never wraps.
- Fader is official 7-bit NRPN parameter `0x17`; official fader/GET/mute/name
  byte vectors are pinned in tests and docs.
- Channel selector ranges cover inputs, groups, auxes, matrices, FX, mains,
  DCA, mute groups and UFX.
- TCP parser handles fragmentation and MIDI running status.
- Readback paths require two fresh stable post-request responses; stale state
  cannot satisfy them.
- Normal clients have no mutation authority and no constructor flag can grant
  it. Generic raw MIDI and normal fader/mute setters are rejected. Scene,
  preamp, phantom, pad, routing, EQ/HPF, sends, pan,
  compressor, gate, delay, polarity, name and colour writes send zero bytes.
- Discovery checks all four official ports and labels results as unverified
  candidates, not verified dLive identity.
- Server, soundcheck, config normalization and both setup UIs preserve
  endpoint/TLS/base-channel/user-profile settings. The C3500 Surface default is
  `192.168.3.71:51328`, MIDI base `11`, user profile `0`; endpoint switching
  selects the matching observed MixRack/Surface address and port. Auto-connect
  no longer silently falls back to base/profile zero.
- A transport-free config preflight and operator-attended read-only artifact
  runner were added. The runner requires the expected dedicated interface and
  refuses to create a TCP client if the macOS route resolves elsewhere.
- `tools/dlive_attended_fader_restore_proof.py` is the only first-write path.
  It requires literal expiring approval, PA/output containment, a muted known
  input, exact name/endpoint/interface binding and a maximum cut of 6 dB. It
  writes WAL send boundaries before transport, restores the exact preimage and
  revokes/disconnects on every exit.
- Before route lookup or TCP, the proof validates one exact owner-controlled
  read-only artifact by path, SHA-256 and content. Endpoint, port, MIDI base,
  user profile, input identity, mute, fader preimage, six GETs, zero-write
  authority and clean teardown must match; fresh 7-bit preimage must match too.
- The literal confirmation is SHA-bound to that exact artifact plus endpoint,
  interface, MIDI base/profile, channel identity, console attestation,
  target/raw value, cut bound and unique approval ID. The former reusable
  generic phrase blocks before route lookup. A durable `prepared` row consumes
  the ID, so a retry requires a fresh ID/token and approval.
- `tools/dlive_validate_fader_proof_plan.py` runs the same artifact validator
  without route lookup or TCP, checks the exact 7-bit bounded cut and emits
  apply/rollback packet hashes plus the required bound confirmation. Its
  receipt explicitly denies execution, physical attestation and authority.
- Emergency interruption after `prepared` or the durable apply boundary closes
  a terminal zero-send WAL record. Interruption after the transport boundary
  invokes exact-preimage rollback and never grants a second apply.
- The restore-only authority compares direction by exact 7-bit fader value and
  enforces the operator's dB cut bound without a tolerance. A target mapping to
  `-inf` or a fresh preimage above `0 dB` blocks before any mutation, preventing
  an unbounded cut or a rollback that would require an unauthorized boost.
- `tools/dlive_recover_fader_restore.py` never replays a target. It can only
  observe an already-restored preimage or restore the exact preimage when fresh
  readback still equals the WAL target; any third value is manual review.

## Network evidence from the attended session

- Dedicated Ethernet: `en7`, link `1000baseT`, DHCP address
  `192.168.3.59/24`; route to `192.168.3.70` selects `en7`.
- `51325` is open; `51327/51328/51329` are closed. This proves a reachable
  plain-TCP MixRack endpoint candidate, not model/firmware/show identity.
- A GET-only search across valid MIDI bases found base `11`. Input 1 name
  replied `KICK`; a follow-up double GET returned stable `KICK`, approximately
  `-0.58 dB`, mute off.
- Read-only MIDI GETs and hardware contact occurred. Mutation attempts and live
  writes were zero; all TCP sessions disconnected cleanly.
- On 2026-08-04 the canonical runner recorded `readback_observed` for
  operator-attested `CDM32+C3500`, firmware `2.0`, Show `lv 1 master`:
  `/Users/dmitrijvolkov/.automixer/runtime/dlive_readonly_probes/20260804T085345Z_04849741-00f0-4d09-8d86-534df1415268.json`, SHA-256
  `f38220bdf85aa606c491ba8ae9323e3db8f2da34ca23bbc356b2161288c93be8`.
  It issued six GETs, zero mutation writes and disconnected cleanly.
- The operator then confirmed that the physical C3500 displayed `KICK`,
  `-0.5827 dB`, mute off. The SHA-bound read-only artifact is accepted without
  granting mutation or live authority.

## Paired-online Surface result and exact next action

Director machine-observed the CDM32 and C3500 paired online on `V2.00 Rev.
83230`. Fresh Surface and MixRack GET-only artifacts matched. The exact approved
Surface attempt then sent one compact packet, but the operator saw no movement
and four readbacks remained at the preimage. Receipt:
`/Users/dmitrijvolkov/.automixer/runtime/dlive_fader_restore_proofs/20260804T141746Z_def89219-2b15-4f4d-a1ef-6309e6bdec8e.json`, SHA-256
`f84d8b0394cfb428a5b4361e7ccca5dca78c3a34083e7cecd9bf8f2b523394f6`.

No automatic or same-plan retry is allowed. Before another mutation:

1. Read back the effective `MixRack / Surface Security` level. Plain TCP
   requires `No Security`; security changes require the affected unit to reboot.
2. Preserve a durable zero-send passive capture or use an Allen & Heath
   known-good client to prove whether this exact console accepts an inbound
   fader SET.
3. If known-good control also fails, stop project writes and correct the console
   configuration. If known-good control works, capture the wire difference and
   fix only that transport difference with offline tests.
4. Only then create a fresh read-only artifact, approval ID and SHA-bound token.

Historical pre-pairing next action follows for audit context:

1. Confirm `Utility / Control / MIDI / Receive MIDI` is ON. Stop if it is OFF
   or unknown.
2. Keep every PA and monitor output physically isolated and Input 1 muted.
3. Confirm the physical C3500 still matches the fresh GET-only Surface artifact
   `20260804T101948Z_54d4cb5b-b850-4644-9ef8-f40918a3394a.json`, SHA-256
   `d4f5dd8ff84d0be7aaf0c5db2352450b59edd8f1170557fa8385288adfc44c40`:
   `KICK`, `-0.582677 dB`, Mute ON.
4. Create a fresh literal restore-only approval bound to that exact path and
   SHA, then apply once to `-3.0 dB`, verify twice, restore the exact preimage,
   verify twice and revoke/disconnect.
5. Preserve the result receipt, WAL path and SHA. Do not grant standing writes.
6. If interrupted, do not start another proof. Run exact-preimage recovery for
   the incomplete WAL transaction; stop on identity or CAS mismatch.

## First restore-only physical attempt — 2026-08-04

The operator physically contained PA/monitor outputs, muted Input 1 and granted
the one-shot literal approval for target `-3.0 dB`. Transaction
`520414f7-4a87-45f5-8e33-e03e449a49a5` sent one apply packet, but the target was
not observed. Four fresh fader readbacks remained at the exact preimage
`-0.5826771653543332 dB`; the WAL closed as
`restored_after_unverified_apply`, `write_state_unknown=false`, and the client
disconnected cleanly. Artifact SHA-256:
`ebb8651a05fba959d2b7a3f1a7d296652ed072e1b51f95f131739b1b7498a1a6`.

The sent full-status NRPN packet was
`BB 63 00 BB 62 17 BB 06 65`. The official table marks the repeated `[BN]`
status bytes as omitted running status, so the encoder is corrected to
`BB 63 00 62 17 06 65`. Offline regression tests pass; a new physical attempt
requires a new literal approval and renewed containment/mute confirmation.

The approved retry sent the corrected packet but again observed the unchanged
preimage four times. Transaction `d64fd0ea-85ad-4db8-abb6-5239b75bbed8` is
terminal with `write_state_unknown=false` and clean teardown; artifact SHA-256
is `8704729a612f57792718f201645b025ae0b99116be25ac15c7e5eed558645323`.
This disproves packet shape as a sufficient root cause. A bounded endpoint
check then found the C3500 Surface at `192.168.3.71:51328`; its read-only probe
matched the MixRack exactly and has SHA-256
`a83964acec2df61a1fc8b84e990030d7579da496a0bd7d8f1aebe0e90b940af7`.
No third write is permitted until `Receive MIDI` is explicitly confirmed ON.
The proof now additionally requires a fresh Surface artifact with `user_profile`
and verifies its exact SHA/content before route lookup or TCP. That artifact
was captured at `20260804T101948Z`, contains base `11`, profile `0`, two fresh
matches per field for `KICK`, `-0.582677 dB`, Mute ON, six GETs, zero mutation
writes and clean disconnect; its SHA-256 is
`d4f5dd8ff84d0be7aaf0c5db2352450b59edd8f1170557fa8385288adfc44c40`.
Emergency
zero-send terminalization and post-send exact rollback are covered by focused
tests; the GET-only capture grants no write or live authority.

## Validation at report-writing time

- Focused dLive transport and safety before the supervised writer checkpoint:
  `107 passed`.
- Restore-only writer/WAL/recovery slice: `28 passed`.
- dLive writer/client slice: `73 passed`.
- dLive/server/config/safety/static integration after final writer hardening:
  `568 passed`.
- dLive/server/config integration: `525 passed`.
- Artifact-binding/emergency proof slice: `41 passed`.
- All dLive tests after this hardening: `110 passed`.
- Frontend production build: passed.
- dLive identity/config pipeline: `475 passed` backend and `27 passed`
  frontend service/UI contracts after base/profile propagation.
- Final expanded dLive/server/config slice: `522 passed`; production frontend
  build passed; six handoff truth rails passed. Final no-WING quick gate:
  `status=passed`, `failed_checks=[]`, `contacts_wing=false`,
  `contacts_backend=false`, `live_write=false`,
  `live_authority_granted=false`.
- Artifact/target-bound confirmation and zero-contact plan-validator slice:
  `46 passed`. The production validator accepted the fresh C3500 artifact for
  `-0.582677 -> -3.0 dB` (`raw 106 -> 101`, cut `2.417323 dB`), apply packet
  `bb630062170665`, rollback packet `bb63006217066a`, with no route/TCP/MIDI or
  execution authority. Candidate approval ID:
  `78121163-b23f-4e84-b830-d9b640f12562`. Required operator token:
  `APPLY_AND_ROLLBACK_DLIVE_FADER_PROOF:sha256:9ab04862d235f2da07a92169ed8178ee3ec6ca42f30614f45205f6bdd342fb5c`.
- Full repository/no-WING rails are intentionally run only after final truth
  surface synchronization; their results must be appended, not predicted.
- Paired-online negative-proof truth preflight: `133 passed` for the focused
  dLive/transport/safety slice. No-WING quick gate artifact
  `artifacts/no_wing_release_check/20260804T143043Z_no_wing_release_check/summary.json`
  is `passed` with `failed_checks=[]`, `environment_blocked_checks=[]`, and no
  WING/backend contact, live write or live authority. Final post-truth
  repetition remains the handoff validator.

Primary protocol source: [Allen & Heath dLive MIDI Over TCP/IP Protocol
V2.0](https://www.allen-heath.com/content/uploads/2024/06/dLive-MIDI-Over-TCP-Protocol-V2.0.pdf).
