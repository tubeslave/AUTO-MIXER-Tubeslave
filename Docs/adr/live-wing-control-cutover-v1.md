# Live WING Control Cutover v1

Date: 2026-09-23
Status: active migration

## Bounded cutover

The real WING fader write family is represented behind the canonical
`backend/live_runtime` control boundary for two target families:
- channel fader (`ch:N / fader_db`);
- main fader (`main:N / fader_db`).

Directors may express slow fader moves as `fader_delta_db`. The control plane
resolves the delta against a fresh physical readback and passes an absolute
`fader_db` action to the WING adapter. This prevents a requested `-0.5 dB`
headroom correction from being misinterpreted as the absolute fader position
`-0.5 dB` when Main is currently somewhere else.

Canonical path:

`Director ProposedAction -> LiveSoundcheckService -> LiveControlPlane -> resolve current+delta -> WingWriteAdapter -> WING OSC -> fresh inbound readback -> VerifiedAction`

This remains deliberately narrower than the legacy Automixer. Unsupported WING
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

## Main-fader semantic finding

The first live `main_headroom_protection` hypothesis used `parameter="fader_db"`
with `value=-0.5`. Hardware adapters interpret `fader_db` as an absolute target.
That means a Main currently at `-6 dB` could have been moved upward to `-0.5 dB`,
the opposite of the intended headroom correction.

The canonical live decision now emits `fader_delta_db=-0.5`. `LiveControlPlane`
reads the current Main position, enforces the proposal's `max_step`, resolves the
absolute target, then writes and verifies that target. A delta larger than its
own declared `max_step` is blocked even in BENCH_TEST. BENCH_TEST bypasses
production policy gates, not the proposal's explicit engineering bound.

## Service composition cutover

`LiveSoundcheckService` owns construction of the new WING control plane for an
active session. The temporary legacy `AutoSoundcheckEngine` is used only to
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
- `backend/wing_addresses.py`: authoritative protocol mapping/ranges used as
  evidence for adapter migration, not as a decision engine.
- `backend/mixer_client_base.py`: shared mixer transport abstraction.
- `backend/live_runtime/control_plane.py`: authoritative mode/write/readback and
  relative-action resolution boundary.

### ADAPT
- WING parameter translation is moving one family at a time into
  `backend/live_runtime/wing_adapter.py`.
- `backend/auto_soundcheck_engine.py` remains a temporary source of discovery,
  audio-capture and physical mixer connection plumbing. Its musical policies are
  not canonical.
- Migrated write family: faders for channels and Mains. Channel EQ remains
  unsupported and therefore fails closed.

### ARCHIVE, not yet eligible
- `backend/auto_fader.py`
- `backend/auto_fader_hybrid.py`
- `backend/live_shared_mix.py` decision policy, including legacy `MasterFaderMove`
  planning
- legacy AutoFOH fader/master decision paths

The repository still contains runtime/import references to `MasterFaderMove` in
`live_shared_mix.py`, `auto_soundcheck_engine.py` and `autofoh_safety.py`.
Therefore the legacy master/fader paths are not yet eligible for archive or
removal even though the new transport path exists.

### DELETE_AFTER_PROOF
None in this pass. No legacy fader/master module is deleted merely because the
new adapter/control-plane path exists. Deletion requires zero runtime/import
references plus replacement tests and BENCH_TEST/HIL evidence.

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
- BENCH_TEST channel-fader write -> query -> verification round trip;
- Main fader write -> query -> verification round trip;
- Director `main_headroom_protection` delta resolving from the current Main
  position (`-6.0 -> -6.5`, never absolute `-0.5`);
- max-step rejection for an oversized relative fader move;
- timeout when no physical/readback callback arrives;
- transport write failure propagation;
- rejection of unsupported parameter/target/range;
- rejection of unresolved delta writes at the hardware adapter boundary;
- service-owned BENCH_TEST proposal -> control plane -> WING adapter round trip;
- service-owned OBSERVE proposal performs read/audit but no mutation;
- missing physical WING transport fails closed;
- non-WING live session cannot accidentally use the WING adapter.

The focused `Stem Offline Test` workflow includes `tests/test_live_runtime_service.py`,
`tests/test_live_control_plane.py`, `tests/test_live_wing_adapter.py`, and the live
handler/decision tests.

## CI note

The prior NumPy 2 studio CI blocker (`numpy.trapz`) was closed on the branch
before this cutover slice. CI for the new Main/delta commits must still pass
before this slice is considered replacement evidence.

## Next cutover

1. Run a physical BENCH_TEST against WING using one deliberately small fader
   delta and retain before -> write -> fresh readback -> verification -> rollback
   evidence.
2. Only after that HIL gate passes, sever runtime references that grant legacy
   AutoFader/master-fader policy direct WING write authority.
3. Migrate channel EQ next, with an explicit band/frequency/Q locator in the new
   action contract before any EQ proposal can reach the hardware adapter.
