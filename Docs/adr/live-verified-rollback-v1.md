# Live verified rollback v1

Date: 2026-09-23
Status: implemented in software, physical WING HIL still required

## Decision

`backend/live_runtime/control_plane.py` owns rollback for actions written by the new live architecture.

A rollback is not a new musical decision. It restores the fresh pre-write value captured by a previous reversible `VerifiedAction` and must use the same hardware adapter/readback boundary as the original write.

Canonical sequence:

`VerifiedAction -> fresh current read -> rollback authorization -> absolute restore write -> fresh readback -> verify -> audit`

## Safety contract

- OBSERVE, PROPOSE and FREEZE never write a rollback.
- Manual freeze blocks rollback in every mode, including BENCH_TEST.
- Write-capable modes may restore a captured value even when the original parameter would be blocked as a new AUTO_SAFE proposal. This permits recovery from an earlier BENCH_TEST or supervised change after returning to production protections.
- Non-reversible actions have no rollback value and fail closed without a transport write.
- Relative proposals such as `fader_delta_db` and `eq_gain_delta_db` are restored using the captured absolute pre-write value.
- PEQ rollback keeps the original `EqBandLocator`, so the WING adapter re-validates the physical frequency/Q fingerprint before restoring gain.
- A transport success is not a successful rollback until fresh readback matches the captured target within the control-plane tolerance.

## Migration impact

### KEEP_CORE
- existing WING/OSC send, subscribe and callback transport;
- physical readback semantics;
- explicit mode/freeze contract.

### ADAPT
- `LiveControlPlane` now owns write rollback and verification;
- `LiveSoundcheckService.rollback_action()` exposes the same canonical boundary to the runtime.

### ARCHIVE
Legacy AutoFOH rollback/action-evaluation code remains comparison material only. It is not imported as the new rollback authority.

### DELETE_AFTER_PROOF
None from this step. Legacy write paths still have runtime references and physical WING HIL has not yet proved the replacement loop.

## Evidence

Focused tests cover:
- relative fader write followed by absolute verified restoration;
- restoration of a previously BENCH_TEST-only/high-risk parameter after entering AUTO_SAFE;
- manual freeze blocking rollback;
- non-reversible action failing closed;
- service-owned WING write/readback/rollback/readback roundtrip.

## Remaining HIL gate

On an explicitly declared BENCH_TEST session with a real WING:
1. capture the current channel fader;
2. apply one bounded reversible move;
3. verify fresh physical readback;
4. invoke `rollback_action`;
5. verify the original value by a second fresh physical readback;
6. repeat once for located PEQ gain while confirming the F/Q fingerprint before rollback.

Only after this evidence, plus severed legacy runtime references, may old fader/EQ rollback/write authorities move to archive.
