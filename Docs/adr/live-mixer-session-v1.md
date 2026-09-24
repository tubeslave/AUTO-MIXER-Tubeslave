# Live mixer session v1

## Status

Accepted for migration. Canonical mixer discovery/physical connection ownership is extracted into `backend/live_runtime/mixer_session.py`; `LiveSoundcheckService` cutover is the next bounded migration step. No legacy module is deleted by this ADR.

## Context

The renovated LIVE/SOUNDCHECK architecture already owns the audio stream, feature bridge, PATCH_VERIFY gate, Directors/Critics and verified WING control path in `backend/live_runtime`. One remaining infrastructure dependency is that `AutoSoundcheckEngine` still owns network mixer discovery and the physical WING/dLive client lifecycle.

The legacy engine mixes useful transport plumbing with heuristic channel classification, fixed musical presets, AutoFOH decision policy, evaluation/rollback policy and its own safety authority. Reusing that class as the long-term connection owner would keep the new runtime coupled to policy that is deliberately being retired.

## Decision

Introduce `LiveMixerConfig`, `LiveMixerTarget`, `LiveMixerStatus` and `LiveMixerSession` in `backend/live_runtime/mixer_session.py`.

`LiveMixerSession` owns only:

- explicit mixer target validation;
- optional discovery through the existing `mixer_discovery` primitives;
- construction of the physical WING or dLive client;
- connect/disconnect lifecycle;
- audit evidence for successful and failed lifecycle transitions.

It contains no musical decisions, channel classification, EQ/dynamics/fader policy, AutoFOH heuristics or control writes beyond the transport client's own connect/disconnect lifecycle.

### Fail-closed rules

1. An unknown mixer type is rejected instead of silently defaulting to dLive.
2. An incomplete target with discovery disabled is rejected.
3. A discovery result that contradicts an explicitly requested mixer type is rejected.
4. Failed or inconsistent connection establishment releases the partially-created client.
5. Start is idempotent while connected and stop disconnects the physical client at most once.

Explicit `mixer_type + mixer_ip` is authoritative and skips discovery. Discovery is used only to complete an otherwise incomplete target. This keeps a configured show network deterministic while retaining discovery as a validated setup primitive.

## Legacy audit / migration classification

### KEEP_CORE

- `backend/mixer_discovery.py` protocol probes and `DiscoveredMixer` data contract.
- `backend/wing_client.py` physical WING transport.
- `backend/dlive_client.py` physical dLive transport.
- Existing verified routing/readback primitives that sit above those clients.

### ADAPT

- `backend/live_runtime/mixer_session.py` is the canonical owner for discovery and physical connection lifecycle.
- `AutoSoundcheckEngine._discover_mixer()` / `_connect_mixer()` remain temporary compatibility seams until `LiveSoundcheckService` owns `LiveMixerSession` and legacy code consumes a non-owning external transport.
- Any read-only legacy observation wrapper is temporary migration plumbing only, not a decision authority.

### ARCHIVE

After service cutover, runtime/import references are severed, and CI + HIL prove the replacement path, archive the legacy discovery/connection orchestration together with the rest of the historical `AutoSoundcheckEngine` implementation. Preserve it for reference without runtime imports.

### DELETE_AFTER_PROOF

None in this step. No transport module is eligible for deletion yet.

## Required proof before severing legacy ownership

The next migration step must prove this lifecycle:

`LiveMixerSession.start -> LiveAudioCaptureSession.start -> legacy non-owning compatibility bind -> LiveAudioCaptureBridge/PATCH_VERIFY -> legacy compatibility stop -> audio stop -> mixer disconnect`

The proof must also show that the physical WING client used by `WingWriteAdapter` is the session-owned client and that legacy shutdown cannot disconnect it early. BENCH_TEST or production write policy remains owned exclusively by `live_runtime`; transport ownership must not resurrect legacy heuristic write authority.

## Tests

`tests/test_live_mixer_session.py` covers explicit WING connection ownership, discovery completion for dLive, discovery/type conflict fail-closed behavior, incomplete-target rejection, failed-connect cleanup, idempotent lifecycle and config validation. The suite is included in the focused `Stem Offline Test` workflow.
