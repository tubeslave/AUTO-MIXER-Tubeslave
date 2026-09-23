# Live WING Control Cutover v1

Date: 2026-09-23
Status: active migration

## Bounded cutover in this pass

The first real WING write family is now represented behind the canonical
`backend/live_runtime` control boundary: channel fader (`ch:N / fader_db`).

New path:

`Director ProposedAction -> LiveControlPlane -> WingWriteAdapter -> WING OSC -> fresh inbound readback -> VerifiedAction`

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

## Migration classification

### KEEP_CORE
- `backend/wing_client.py`: WING handshake, OSC send/receive, subscriptions and
  transport state. No decision policy is added.
- `backend/mixer_client_base.py`: shared mixer transport abstraction.
- `backend/live_runtime/control_plane.py`: authoritative mode/write/readback
  boundary.

### ADAPT
- WING parameter translation is being moved one family at a time into
  `backend/live_runtime/wing_adapter.py`.
- First migrated family: channel fader.

### ARCHIVE, not yet eligible
- `backend/auto_fader.py`
- `backend/auto_fader_hybrid.py`
- legacy AutoFOH fader decision paths

They still require runtime/import reference severing and replacement coverage.
They must not receive new decision features during migration.

### DELETE_AFTER_PROOF
None in this pass. No legacy fader module is deleted merely because the new
adapter exists. Deletion requires zero runtime/import references plus replacement
tests and BENCH_TEST/HIL evidence.

## Mode behavior

`BENCH_TEST` may authorize broad writes for development visibility, but adapter
support remains explicit. A mode bypass is not a protocol bypass: an action that
has not been migrated into `WingWriteAdapter` fails closed.

Production modes continue through `LiveControlPlane` authorization and readback.

## Verification gate

Automated tests cover:
- fresh callback-based WING fader readback;
- BENCH_TEST fader write -> query -> verification round trip;
- timeout when no physical/readback callback arrives;
- transport write failure propagation;
- rejection of unsupported parameter/target/range.

The focused CI workflow includes `tests/test_live_wing_adapter.py`.

## Next cutover

After CI is green for this slice, connect `WingWriteAdapter` to the live service
composition root and route one real BENCH_TEST fader proposal through it. Then
collect physical WING HIL evidence before severing any legacy fader runtime path.
