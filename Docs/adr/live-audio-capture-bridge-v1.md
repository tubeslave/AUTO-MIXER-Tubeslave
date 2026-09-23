# ADR: Live AudioCapture bridge v1

Status: accepted for migration validation

## Context

The canonical LIVE/SOUNDCHECK implementation lives in `backend/live_runtime`, while the repository already has a validated multichannel device/ring-buffer transport in `backend/audio_capture.py`. Replacing that transport during the renovation would add risk without improving decision authority. At the same time, running FFT analysis, Directors, Critics or WING writes directly inside the sounddevice callback would make the audio callback vulnerable to analysis latency.

The previous USB feature-ingress ADR established two hard constraints: the MVP capture contract is exactly 48 channels at 48 kHz, and USB input stems must never be summed into a synthetic Main. Main evidence must come from an explicit physical meter/readback or real Main tap and be time-coherent with the captured window.

## Decision

Add `backend/live_runtime/capture_bridge.py` as the migration boundary between legacy capture transport and the canonical runtime.

`LiveAudioCaptureBridge`:

1. accepts only a 48-channel / 48 kHz `AudioCapture`-compatible source;
2. subscribes through the existing named subscriber API without owning or stopping the audio device;
3. snapshots the most recent equal-length window from all 48 ring buffers at the existing post-write callback boundary;
4. performs only the bounded raw-memory copy on the capture callback thread;
5. sends FFT feature extraction, Main-evidence assembly and `LiveSoundcheckService.process_feature_snapshot()` to one dedicated non-realtime worker;
6. maintains a single pending snapshot, replacing old pending work instead of allowing an unbounded realtime-analysis backlog;
7. requires an injected `MainEvidenceProvider(capture_timestamp)` and rejects missing or stale Main evidence before the service can make a decision;
8. forwards an explicit channel-role mapping as configuration, not as a legacy decision policy;
9. records bridge health counters for captured/processed/replaced/incomplete/missing-Main/failure cases.

No legacy `auto_*` decision module is imported by this path.

## Legacy classification

### KEEP_CORE

`backend/audio_capture.py`: retain device access, sounddevice callback, per-channel ring buffers, and named subscribe/unsubscribe transport. No new musical decision feature is added to this module.

### ADAPT

`backend/live_runtime/feature_stream.py`: existing typed USB evidence extractor and `assemble_mix_features()` are composed behind the bridge.

`backend/auto_soundcheck_engine.py`: remains a temporary migration bridge only where current startup/discovery code still owns the existing `AudioCapture` instance. Its soundcheck heuristics are not reused as LIVE decision authority.

### ARCHIVE

Legacy AutoFader/AutoEQ/AutoFOH feature-to-action policies remain archive candidates once runtime/import references are severed and replacement/HIL evidence exists. Their policies are not called from the capture bridge.

### DELETE_AFTER_PROOF

None in this step. No legacy module is deleted before reference severing plus replacement tests/HIL.

## Safety consequences

- No physical WING write is introduced by the bridge itself. Production/BENCH_TEST write rules remain entirely in `LiveControlPlane`.
- Missing, stale or non-finite Main evidence fails closed before a feature snapshot reaches the Director.
- Incomplete 48-channel windows are dropped rather than padded across channels.
- Analysis backlog is bounded to one latest snapshot.
- The audio transport lifecycle stays independent: stopping the bridge unsubscribes it but does not stop `AudioCapture`.

## Validation gate

Focused tests must prove:

- rejection of non-48-channel or non-48-kHz capture;
- 48-channel snapshot forwarding into the canonical service;
- feature/service work runs on a worker thread rather than the capture callback thread;
- missing Main evidence cannot reach the service;
- stale Main evidence cannot reach the service;
- incomplete windows are dropped;
- stop unsubscribes without stopping capture transport.

A real WING HIL write is not part of this ADR and remains gated by an explicitly declared BENCH_TEST session.

## Next migration target

Provide an authoritative Main-bus evidence provider from physical WING metering or a configured post-console Main audio tap, then compose bridge lifecycle into the canonical live runtime startup/FSM. This must preserve the same timestamp-coherence and fail-closed contract.
