# Live Service Mixer Ownership Lifecycle v1

Date: 2026-09-24
Status: accepted for R2/R3 migration; physical WING HIL still required before legacy archival

## Context

`LiveMixerSession` already owns validated mixer target resolution, client construction, physical connect and disconnect. `LegacyExternalMixerSeam` already exposes that connected transport to `AutoSoundcheckEngine` through a deliberately read-only proxy while bypassing the legacy engine's own discovery/connect methods.

Before this decision, `LiveSoundcheckService` still let `AutoSoundcheckEngine` own mixer discovery/connection even though audio capture had already moved to canonical `live_runtime` ownership. That left two generations of lifecycle authority in the active LIVE/SOUNDCHECK composition.

## Decision

For explicitly configured canonical live sessions, identified by a configured `LiveCaptureBridgeConfig`, `LiveSoundcheckService` now owns both physical hardware lifecycles:

1. `LiveMixerSession.start()` resolves and connects the mixer once.
2. `LegacyExternalMixerSeam.bind()` gives the frozen legacy engine a non-owning read-only proxy and proof-only discovery/connect bypasses.
3. `LiveAudioCaptureSession.start()` opens the 48-channel capture once.
4. `LegacyExternalAudioCaptureSeam.bind()` gives the legacy engine a non-owning capture proxy.
5. `AutoSoundcheckEngine.start_async()` remains temporarily only as compatibility lifecycle/event plumbing. Its legacy mixer discovery/connect and audio-start ownership paths are bypassed.
6. `LiveAudioCaptureBridge` attaches to the canonical capture and drives PATCH_VERIFY and the canonical Directors/Critics.

Teardown order is deliberately the inverse ownership chain:

`bridge stop -> legacy compatibility stop -> audio seam detach -> audio session stop -> mixer seam detach -> mixer session stop`

The seams stay bound while legacy `stop()` runs. Therefore legacy calls to stop/disconnect cannot close physical resources. Physical teardown is performed exactly once by the canonical sessions.

## Control authority

The authoritative WING control path uses the physical client owned by `LiveMixerSession`, never the legacy read-only proxy. `LiveSoundcheckService._active_wing_transport()` resolves to the canonical physical WING client when mixer ownership is active.

`BENCH_TEST` does not change this ownership rule. It may relax production write restrictions only inside the canonical `live_runtime` control plane so decisions can be observed on the console. It does not grant legacy heuristic code write authority.

For a declared real soundcheck/concert/show, normal production protections remain mandatory.

## Compatibility path

Sessions without canonical capture composition retain the previous `AutoSoundcheckEngine` discovery/connect/audio ownership temporarily. This is a compatibility seam only. No new decision features may be added to that legacy path.

## Repository structure

The pre-existing service implementation is retained as `backend/live_runtime/service_core.py`. `backend/live_runtime/service.py` is the canonical facade that adds single-owner hardware composition while preserving the existing control-plane, PATCH_VERIFY, Director/Critic and iteration contracts. This is internal layering, not a second runtime authority: external callers continue importing `backend.live_runtime.service`.

## Migration classification

### KEEP_CORE

- `backend/mixer_discovery.py` probes and validated discovery primitives.
- `backend/wing_client.py`, `backend/dlive_client.py` transport primitives.
- `backend/audio_capture.py`, `backend/audio_device_scanner.py` stream/device primitives.
- routing/readback/address/metrics primitives already classified as infrastructure.

### ADAPT

- `backend/live_runtime/mixer_session.py` as canonical physical mixer lifecycle owner.
- `backend/live_runtime/audio_capture_session.py` as canonical physical audio lifecycle owner.
- `backend/live_runtime/legacy_mixer_seam.py` and `legacy_audio_capture_seam.py` as temporary migration seams.
- `backend/live_runtime/service.py` as the canonical composition/lifecycle owner.
- `AutoSoundcheckEngine` only for the smallest remaining compatibility lifecycle/event surface until it is severed.

### ARCHIVE after proof

- `AutoSoundcheckEngine._discover_mixer()` and `_connect_mixer()` ownership/orchestration.
- `AutoSoundcheckEngine._start_audio()` ownership/orchestration.
- the legacy soundcheck FSM, instrument presets and AutoFOH/AutoFader/AutoEQ decision/evaluation/rollback authorities once runtime/import references are severed and replacement HIL evidence passes.

### DELETE_AFTER_PROOF

No new candidate is promoted by this pass. Nothing is deleted.

## Automated evidence

`tests/test_live_service_capture_bridge_lifecycle.py` now proves on the migrated composition that:

- mixer physical ownership starts before audio and legacy compatibility plumbing;
- legacy discovery/connect methods are not invoked;
- legacy mixer client slots receive a proxy rather than the physical client;
- verified control transport resolves to the physical canonical WING client;
- there is one physical audio stream and one physical mixer connection;
- teardown order is `bridge -> legacy -> audio -> mixer`;
- physical audio stop and mixer disconnect occur exactly once;
- bridge startup failure does not fall back to legacy hardware ownership;
- legacy-start failure cleans both canonical physical owners;
- the unconfigured compatibility path still behaves as before.

The focused `Stem Offline Test` live-runtime decision/composition suite passed on implementation commit `f4ef2098f79b5f9b98ee687f268f26d4efce2cd2` before this documentation update.

## HIL gate before archival

Do not archive or remove the legacy mixer/audio ownership paths until a real WING HIL session proves all of the following:

1. exactly one physical WING connection is established;
2. exactly one 48-channel audio capture is established;
3. fresh routing readback and native Main meter PATCH_VERIFY reach `LISTEN` with the intended console patch;
4. canonical Director/Critic control writes and readback verification use the physical `LiveMixerSession` client;
5. legacy proxy attempts cannot mutate or disconnect the console, including during BENCH_TEST;
6. stop/restart cycles disconnect and reopen the physical mixer/audio resources exactly once without stale subscriptions or duplicate clients.

## Current migration state

R2 extraction and R3 LIVE cutover now have canonical ownership for both mixer transport and audio capture in the configured LIVE/SOUNDCHECK path. `AutoSoundcheckEngine` is no longer required to own either physical resource in that path. Its remaining active role is compatibility lifecycle/event plumbing plus still-referenced legacy surfaces outside this composition.

The next bounded migration goal is to audit what `LiveSoundcheckService` still consumes from `AutoSoundcheckEngine.start_async()/stop()` and its callbacks, extract the minimum useful lifecycle/event primitive into `backend/live_runtime`, and sever the engine itself from the canonical configured path without reviving its heuristic FSM or policy.