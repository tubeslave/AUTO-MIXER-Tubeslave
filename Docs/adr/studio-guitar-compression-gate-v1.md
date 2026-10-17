# ADR: Guitar Compression Gate v1

## Decision

Rhythm/electric guitar compression is **no-change first**. Before proposing another compressor timing, the STUDIO Director must demonstrate an actionable local-dynamics problem on the current processed guitar. Distorted/self-limited material is not compressed merely because an insert exists in the historical recipe.

## Belye Stai source-bound boundary

For the frozen Belye Stai recipe the local GTR chain is reproduced as:

`RAW -> HP 90 Hz -> LP 9 kHz -> bell 310 Hz -1.8 dB Q0.8 -> bell 3 kHz -0.8 dB Q0.8 -> compressor -> frozen final level`

The delivered compressor uses ratio 2:1, 25 ms attack, 180 ms release, 5 dB knee and 2.8 dB maximum GR. Replacement occurs at that original insert, never after an already processed guitar. Frozen controls are bound to exact source and pre-compression hashes.

Independent reproduction against the archived procedural recipe on the real 207 s GTR source is sample-identical: max absolute error 0.0 and processed float-PCM SHA-256 `1f163ea07f33ee266bb126f0a1225fe17822f7b6a6835b7509075dad786cc375`.

## Actionability gate

Evidence is measured on fixed, reusable windows:

- 40 ms RMS / 20 ms hop for active guitar level;
- median P90-P10 active-level spread inside 2 s blocks as the local consistency proxy;
- P90-P10 spread of active block means as a macro-dynamics guard;
- fixed high-band pick-event windows for attack/body contrast.

Predeclared v1 policy requires at least 12 active blocks and 24 pick events and a median local spread of at least 2.8 dB before compressor changes are even opened. If actionable, only bounded attack/release probes are proposed; threshold, ratio, knee and max GR remain frozen. A candidate must improve local spread by at least 0.10 dB while losing no more than 0.25 dB pick attack/body contrast and changing macro spread by no more than 0.20 dB.

These are technical proxies, not a music-quality score. Any technical survivor still requires complete routed-session rerender and level-matched human listening.

## Real Belye Stai result

On the full 207 s source the baseline evidence contained 83 active 2 s blocks and 491 pick events. Median local spread was **2.6685 dB**, below the predeclared 2.8 dB actionability threshold; macro spread was 3.0473 dB and median pick attack/body was 1.4595 dB. Therefore the gate returned `no_change` and generated **zero compressor candidates**.

The threshold was not lowered after seeing the result. No full-session rerender or A/B was produced because there is no candidate to audition. The current guitar remains unchanged.

## Boundaries

No accepted vocal, bass/drum adapters, mastering, live code, common compressor core, or Perceptual Critic thresholds are changed. No neural audio processing or paid external credits are used. Human listening remains mandatory for subjective acceptance of any future guitar survivor.
