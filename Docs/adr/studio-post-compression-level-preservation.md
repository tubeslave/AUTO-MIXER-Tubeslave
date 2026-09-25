# Post-compression source level preservation — Belye Stai v4 audit

User likes the v4 compression but hears quieter kick/snare, not toms. Do not change the compressor envelopes to repair a gain problem.

## Reproduction

Exact 18 original WAVs and the saved runtime at 326cec43286ea3722608fbc92de7dfe0c0481692 were used locally. Reconstructed v2 and v4 premaster files are byte-identical to the actual deliveries:
- v2: 73e407ac2893b94ea49aecfa4274da52c3131cc0dafb04fc394c0f94a596eb59
- v4: b3609a1ad465a53aeef4af8de4f6ca71af99077803a1e3364ae8422ef663f776
All six reconstructed override-source PCM hashes match v4 candidate_report.json.

## Findings, independently measured

The adapters intentionally freeze the previous static output gain to isolate the compressor intervention. The v4 song delivery did not add a separate fixed-reference level match afterwards. On the SAME reference-active 20 ms frames:
- KICK q92: -0.1165532763 dB;
- SNARE q94: -0.2378898615 dB;
- TOM_1/TOM_2/FLOOR q98: +0.1622/+0.1633/+0.1810 dB.
This is not a multi-dB collapse and RMS is not a perceptual loudness verdict.

Instrumenting the exact full-session gain traces (without changing any returned audio) gives dry post-bus source deltas after matching whole-mix LUFS: KICK -0.068741 dB, SNARE -0.187939 dB. These are actual-render contributions, not baseline-minus-dry substitution. Sum reconstruction error of the dry DRUMS bus is below 1e-7 linear.

FFmpeg remeasurement of delivered masters: v2 -15.13 LUFS; v4 -15.57 LUFS. Both WAV true peaks are -1.62 dBTP. Thus the master is also about 0.44 LU quieter globally. The saved v4 search scored loudness before final static true-peak attenuation (-0.395858 dB); its selected output was not brought back to the v2 delivery loudness. Final loudness must be evaluated AFTER every safety attenuation and after file/codec export.

## Bounded remediation

Add explicit match_processed_source: fixed previous-processed-source activity mask, channel-power RMS, one bounded constant gain, no alteration to attack/release/ratio/knee or timeline. Silence, shape mismatch and excessive gain fail closed. No automatic peak normalization or hidden gain cap is allowed to masquerade as a match. The helper does not replace or retune frozen adapters and does not claim full-session headroom.

For this song: KICK +0.1165532763 dB and SNARE +0.2378898615 dB AFTER local compression, BEFORE the full original session graph. These corrected arrays equal the helper's output sample-for-sample. Vocal, toms and all other input arrays/settings remain unchanged from v4. Rerender every dependent bus, effect and mastering stage. Post-bus remaining matched level error vs v2: -0.044405/-0.044844 dB, within 0.05 dB. Do not infer that all other buses remain byte-identical after shared compression.

## Acceptance boundaries

10 new local tests passed. Full PR CI pending. This additive helper is explicitly callable; it does not pretend to retrofit every historical song script. The corrected song uses the measured gain and is an audition until human listening. Final audio, codec peaks, limiter GR and crest budget still require validation. Never lower safety limits to force a loudness match. No live/backend changes, no neural audio, no paid external credits, no original or old-delivery writes.
