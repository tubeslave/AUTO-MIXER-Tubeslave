# ADR: Belye Stai Local Vocal Processor Adapter v1

## Context
The accepted Belye Stai SessionRenderer starts after local track processing. Routed Compression Family v1 therefore tested extra compression after the original vocal dynamics. That cannot answer whether Compression Director can improve the original compressor decision.

## Decision
Add `audio_workbench/belye_stai_vocal.py`, a song/version-specific deterministic adapter for raw `VALERA_VOX`. It reproduces the frozen delivery order: HPF/LPF and three bells, bounded expander, slow ride, first compressor, second compressor, de-esser and final active-RMS level stage.

The adapter exposes the signal immediately before the first compressor. With no override it executes the frozen delivery compressor. With `first_stage_config` it replaces only that compressor with the shared `LinkedCompressor`; it does not append another compressor downstream. The second compressor, de-esser and level stage remain frozen so one causal intervention is tested at a time.

No-change output remains non-promotable because this is a reproduction boundary, not an artistic acceptance. Any replacement sets mandatory human-review/listening metadata and remains ineligible for autonomous baseline promotion.

## Real reproduction proof
Using the original `18_VALERA_VOX.wav`, 44.1 kHz, 9,128,700 samples, the legacy procedural chain from the delivered `mix_dsp.py` and the new adapter produced:

- pre-compression float PCM: max absolute error 0.0, SHA-256 `0925bca62f2c44b78ac999c1522697228d116681d1dc0390d047dc2c335cfb2f`;
- fully processed local vocal: max absolute error 0.0, sample-for-sample equality, SHA-256 `64123ff2668b9f3ebf9ef4657c3af2416a4939abecb5972448caffa994df8d26`;
- raw input unchanged.

Baseline first-compressor measured values on that source were threshold -37.044899 dBFS, max GR 4.499971 dB and sampled p95 GR 3.499265 dB. The frozen second stage measured threshold -37.833474 dBFS, max GR 2.443368 dB and sampled p95 GR 1.345340 dB.

## Consequence
The next experiment can generate Compression Director v1.1 candidates from the actual pre-first-compressor signal, replace that compressor in place, then pass the resulting processed vocal into the exact Belye Stai SessionRenderer and existing Perceptual Critic/Autonomous Iteration gates. Human listening still decides subjective acceptance.