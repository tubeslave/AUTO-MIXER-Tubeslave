# ADR: Legacy external mixer seam v1

Date: 2026-09-24
Status: accepted, active migration seam

## Context

`backend/live_runtime/mixer_session.py` now owns canonical mixer discovery and physical connect/disconnect lifecycle, but `AutoSoundcheckEngine` still participates in the temporary compatibility path and historically owns `_discover_mixer()`, `_connect_mixer()`, `mixer_client` and `_real_mixer_client`.

Passing the physical client into the legacy engine directly would recreate two authorities over one console. It would also leave a path for old heuristic AutoFOH code to call mixer write methods, including in a development BENCH_TEST session. BENCH_TEST is allowed to relax production write restrictions only inside the canonical live_runtime control path. It must not resurrect legacy decision authority.

## Decision

Add `backend/live_runtime/legacy_mixer_seam.py` as a temporary migration boundary.

`LiveMixerSession` remains the sole owner of the physical client and its discovery/connect/disconnect lifecycle. `LegacyExternalMixerSeam` binds a read-only proxy into both legacy client slots and temporarily replaces legacy `_discover_mixer()` and `_connect_mixer()` with proof-only bypasses. The bypasses succeed only while the canonical session remains connected to the same client/target and both legacy client slots still contain the expected proxy.

The proxy permits read-style `get_*`, `read_*` and `query_*` calls plus a small set of scalar connection metadata. `connect()`, `disconnect()`, `set_*`, generic `send`, unknown methods and other non-read access fail closed. The underlying physical client is not exposed by a public proxy property.

Binding also mirrors only compatibility metadata (`mixer_type`, `mixer_ip`, `mixer_port`) from the canonical target. Detach restores the exact prior instance state and refuses to overwrite a client slot that was changed while external ownership was active.

This seam does not perform any console mutation, does not add a new decision policy, and does not change BENCH_TEST rules. BENCH_TEST may make canonical `live_runtime` decisions visible on the console through the verified control plane, but the legacy seam remains read-only in every mode.

## Migration classification

### KEEP_CORE

- `backend/mixer_discovery.py` probe primitives.
- validated `WingClient` / `DLiveClient` transport, routing and readback primitives.

### ADAPT

- `backend/live_runtime/mixer_session.py` as canonical physical mixer lifecycle owner.
- `backend/live_runtime/legacy_mixer_seam.py` as temporary compatibility plumbing only.

### ARCHIVE after proof

- legacy `AutoSoundcheckEngine._discover_mixer()` / `_connect_mixer()` ownership and orchestration after `LiveSoundcheckService` owns `LiveMixerSession`, runtime references are severed, replacement tests pass and HIL proves the new lifecycle.
- legacy heuristic soundcheck FSM, instrument presets and AutoFOH/AutoFader/AutoEQ decision/evaluation/rollback paths after their respective replacement proofs.

### DELETE_AFTER_PROOF

No new candidate is promoted by this change.

## Validation gate

Software evidence must prove:

1. binding does not invoke a second discovery or physical connect;
2. read methods delegate to the live-owned client;
3. lifecycle and write methods are blocked before reaching the physical client;
4. ambiguous pre-existing legacy ownership fails closed;
5. loss/replacement of the canonical client or proxy fails closed;
6. detach restores legacy state without disconnecting the physical client;
7. the focused live-runtime CI suite includes the seam tests.

Physical HIL remains required before legacy mixer lifecycle code is archived. No physical WING operation is part of this ADR implementation.

## Next cutover

Wire `LiveMixerSession` and `LegacyExternalMixerSeam` into `LiveSoundcheckService`. The service should start the canonical mixer session before compatibility startup, bind the legacy seam without transferring ownership, and tear down the physical mixer only after bridge/legacy compatibility teardown and seam detach. Only after that lifecycle is covered by tests and WING HIL may the legacy discovery/connect ownership move from ADAPT to ARCHIVE.
