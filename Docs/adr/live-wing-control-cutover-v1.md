# Live WING Control Cutover v1

Date: 2026-09-23
Status: active migration

## Bounded cutover

The first real WING write family is represented behind the canonical
`backend/live_runtime` control boundary: channel fader (`ch:N / fader_db`).

Canonical path:

`Director ProposedAction -> LiveSoundcheckService -> LiveControlPlane -> WingWriteAdapter -> WING OSC -> fresh inbound readback -> VerifiedAction`

This is deliberately narrower than the legacy Automixer. Unsupported WING
parameters fail closed until they receive their own adapter mapping and tests.

## Critical readback finding

`backend/wing_client.py` maintains a local state cache and optimistically updates
that cache when a command is sent. Therefore, reading `WingClient.state`
immediately after a write does not prove that the physical console accepted the
command.

`WingWriteAdapter` does not use that cache for verification. It subscribes to
the requested OSC address, sends an explicit query, and requires a fresh inbound
WING callback before returning a value to `LiveControlPlane`. Missing readback
raises a timeout instead of being silently accepted.

This preserves the existing, proven WING transport while moving verification
semantics into the canonical live runtime.

## Service composition cutover

`LiveSoundcheckService` now owns construction of the new WING control plane for
an active session. The temporary legacy `AutoSoundcheckEngine` is used only to
obtain the physical WING transport while discovery/capture orchestration is
still being migrated.

Rules:
- prefer `engine._real_mixer_client` over an observation wrapper;
- never use an `ObservationMixerClient` as proof of physical readback;
- only WING mixer types may create `WingWriteAdapter` today;
- unsupported mixer types fail closed until their adapter exists;
- starting/stopping a session resets control-plane ownership;
- every control-plane decision is retained in a service audit list;
- `get_status()` exposes `control_plane_ready` and `control_audit_count` for HIL visibility.

`LiveSoundcheckService.execute_action()` takes mode exclusively from the active
explicit `LiveStartRequest`. It cannot infer BENCH_TEST merely from a connected
console.

## Migration classification

### KEEP_CORE
- `backend/wing_client.py`: WING handshake, OSC send/receive, subscriptions and
  transport state. No decision policy is added.
- `backend/mixer_client_base.py`: shared mixer transport abstraction.
- `backend/live_runtime/control_plane.py`: authoritative mode/write/readback
  boundary.

### ADAPT
- WING parameter translation is moving one family at a time into
  `backend/live_runtime/wing_adapter.py`.
- `backend/auto_soundcheck_engine.py` remains a temporary source of discovery,
  audio-capture and physical mixer connection plumbing. Its musical policies are
  not canonical.
- First migrated write family: channel fader.

### ARCHIVE, not yet eligible
- `backend/auto_fader.py`
- `backend/auto_fader_hybrid.py`
- legacy AutoFOH fader decision paths

They still have runtime/import references and there is not yet physical HIL
proof for the replacement. They must not receive new decision features during
migration.

### DELETE_AFTER_PROOF
None in this pass. No legacy fader module is deleted merely because the new
adapter/service path exists. Deletion requires zero runtime/import references
plus replacement tests and BENCH_TEST/HIL evidence.

## Mode behavior

`BENCH_TEST` may authorize broad writes for development visibility, but adapter
support remains explicit. A mode bypass is not a protocol bypass: an action that
has not been migrated into `WingWriteAdapter` fails closed.

Production modes continue through `LiveControlPlane` authorization and readback.
OBSERVE/PROPOSE/FREEZE can inspect the current physical value and audit the
proposal, but do not call the WING write operation.

## Automated verification gate

Tests cover:
- fresh callback-based WING fader readback;
- BENCH_TEST fader write -> query -> verification round trip;
- timeout when no physical/readback callback arrives;
- transport write failure propagation;
- rejection of unsupported parameter/target/range;
- service-owned BENCH_TEST proposal -> control plane -> WING adapter round trip;
- service-owned OBSERVE proposal performs read/audit but no mutation;
- missing physical WING transport fails closed;
- non-WING live session cannot accidentally use the WING adapter.

The focused `Stem Offline Test` workflow includes `tests/test_live_runtime_service.py`,
`tests/test_live_control_plane.py`, `tests/test_live_wing_adapter.py`, and the live
handler/decision tests.

## CI note

The latest full repository matrix observed before this service cutover had a
studio-only failure on Python 3.11/3.12: `audio_workbench/mastering/analyzer.py`
uses removed `numpy.trapz` under NumPy 2.4.6. Python 3.10 passed, and the focused
live/stem workflow passed. This is not evidence against the live cutover and is
left to the studio pipeline rather than patched in legacy live work.

## Next cutover

Run a physical BENCH_TEST against WING using one deliberately small channel
fader proposal and retain the audit/readback evidence. Only after that HIL gate
passes should the runtime references that grant legacy AutoFader code direct
WING write authority be severed. The next software migration family after fader
HIL is channel EQ, one explicitly mapped parameter family at a time.
