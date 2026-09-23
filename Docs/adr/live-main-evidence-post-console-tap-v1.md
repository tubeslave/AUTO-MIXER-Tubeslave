# Live Main Evidence: Post-console Tap v1

Date: 2026-09-24
Status: active R3 migration decision

## Decision

The first authoritative Main-bus evidence source for the canonical live runtime is a **real post-console Main audio tap** routed back into reserved WING USB capture slots.

`backend/live_runtime/main_evidence.py` owns the measurement primitive. `PostConsoleMainTapEvidenceProvider` receives the same coherent 48-channel snapshot already copied by `LiveAudioCaptureBridge` and computes Main RMS, peak and crest from the explicitly configured mono/stereo tap only.

The live runtime must never estimate Main level by summing captured input stems.

## Why this path first

WING exposes dedicated meter transport, but that transport is separate from the ordinary OSC property/readback path and requires its own request/renew/data-packet lifecycle. Reusing the existing OSC `WingClient.state` cache as if it were an audio meter would be false evidence.

A routed post-console USB return provides an immediately testable Main signal while preserving the architectural rule that Main evidence represents the processed console output rather than pre-console stems. A future WING meter adapter may implement the same evidence contract without changing Directors/Critics.

## Safety invariants

- exactly one authoritative Main evidence source is configured per `LiveAudioCaptureBridge`;
- Main tap channels are explicitly reserved and excluded from channel-level feature/decision analysis;
- Main evidence and channel features come from the same copied capture snapshot and therefore share one timestamp;
- tap blocks must be finite and exactly 48 channels at the canonical USB ingress;
- stale/missing evidence still fails closed through `assemble_mix_features()` / bridge status;
- the provider performs measurement only. It does not configure routing and it does not make musical decisions;
- routing of the post-console Main tap must be verified separately during PATCH_VERIFY/startup before autonomous writes are permitted.

## Migration classification

### KEEP_CORE
- `backend/audio_capture.py`: device access, callback and ring-buffer transport remain unchanged.

### ADAPT
- `backend/live_runtime/capture_bridge.py`: composes authoritative Main evidence with the feature worker.
- `backend/signal_analysis.py`: RMS/peak/crest/spectral concepts remain useful evidence references, but its compressor-oriented state/policy is not imported into `live_runtime`.

### ARCHIVE
Legacy AutoFOH/AutoCompressor feature-to-action orchestration remains an archive candidate after runtime references are severed and replacement HIL passes.

### DELETE_AFTER_PROOF
None added by this change.

## Verification gate

Focused tests must prove:
1. mono/stereo tap validation and finite-sample rejection;
2. deterministic RMS/peak/crest extraction;
3. exact timestamp coherence with the captured channel frame;
4. reserved Main tap slots do not enter channel-level features;
5. configuring zero or multiple Main authorities fails closed.

Hardware gate remains PATCH_VERIFY/HIL: route a known post-fader Main source into the configured USB tap slots, compare it against the physical WING Main meter, then permit the bridge to enter the autonomous live iteration path.
