# Live USB Feature Ingress v1

Date: 2026-09-23
Status: R3 migration contract

## Decision

The first live audio-analysis ingress is a strict WING USB MVP: 48 input channels at 48 kHz. `backend/live_runtime/feature_stream.py` is the canonical channel-feature boundary. It accepts a `[frames, 48]` floating-point block, validates shape and finite samples, and emits typed `ChannelFeatures` for selected 1-based mixer channels.

The ingress extracts evidence only. It does not choose EQ, compression, fader, routing or any other mixer action. Musical decisions remain in `live_runtime` Directors/Critics and writes remain behind `LiveControlPlane`.

## Main-bus evidence is separate

Raw pre-console USB input stems are not authoritative evidence for post-console Main headroom. The runtime therefore must not sum input channels and call that result Main.

`Usb48FeatureExtractor` emits a channel-only `UsbChannelFeatureFrame`. `assemble_mix_features()` requires explicit `MainFeatureEvidence` from an actual Main meter/readback or a real configured Main audio tap before it can construct `MixFeatures`. Main evidence that is non-finite or more than 250 ms away from the USB frame fails closed by default.

This distinction is important because `main_headroom_protection` is allowed to move the Main fader. Fabricated Main evidence would create a false control authority.

## Migrated primitives

The channel feature implementation keeps only transport/DSP evidence ideas that survive the renovation:

- RMS, sample peak and crest factor from the captured signal;
- Hann-window FFT spectral centroid;
- low-mid/reference energy ratio;
- a normalized harshness evidence proxy using 2-4 kHz plus weighted 4-8 kHz energy;
- activity evidence using the legacy `-50 dB` activity threshold concept.

The old `backend/auto_fader_v2/core/activity_detector.py` is **ADAPT**, not runtime authority. Its useful threshold concept is transferred into the new feature layer; the legacy module is not imported by `live_runtime`.

`backend/audio_capture.py` remains **KEEP_CORE** as capture/ring-buffer/device plumbing. No new decision behavior is added to it.

## Validation and fail-closed rules

The USB MVP rejects:

- sample rates other than 48 kHz;
- capture layouts other than exactly 48 channels;
- zero-length or non-2D blocks;
- NaN/Infinity samples;
- duplicate or out-of-range selected channel numbers;
- non-finite Main evidence;
- stale Main evidence.

Near-silence emits stable level evidence and no spectral evidence instead of meaningless FFT descriptors.

## Not included in this step

This change does not yet subscribe the extractor directly to `AudioCapture` callbacks and does not perform WING writes. The next bounded migration step is a live-runtime capture adapter that obtains coherent 48-channel analysis windows from the KEEP_CORE capture transport and feeds the existing `LiveSoundcheckService.process_feature_snapshot()` only after explicit Main evidence is available.

No BENCH_TEST is inferred from device connectivity. Hardware mutation remains disabled unless an explicit development/test session selects BENCH_TEST.
