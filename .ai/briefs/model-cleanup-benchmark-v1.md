# Model Cleanup Benchmark v1

Previous heuristic cleanup v0.9/v0.10 is rejected by listening because it introduced more defects.

Research findings:
- IRMR/t-UNet is directly relevant to multitrack interference reduction, but its public repository contains training code and no pretrained model weights in the tUNet directory.
- Bleed No More (ICASSP 2026) is highly relevant and reports successful generative interference reduction, but no public inference repository/weights were found in the current search.
- 2026 permutation-equivariant multichannel debleeding work is relevant, but public code/weights were not found in the current search.
- DeepFilterNet has public pretrained weights/binaries, but it is a speech denoiser. It may be benchmarked on vocal stationary noise only; it must not be treated as a musical debleed model.

Benchmark protocol:
1. use 24 s difficult excerpts from Ptitsa, not the whole song;
2. keep synchronized reference mics with each excerpt;
3. every candidate emits RAW, CLEAN and REMOVED;
4. inspect REMOVED for target attacks, consonants, notes and tails;
5. reject any model that improves isolation but audibly damages the intended source;
6. de-noise and de-bleed are scored separately;
7. no gate/expander fallback if model inference is unavailable.

Next implementation target:
- adapter interface for pretrained model inference;
- residual-blend control 0..1;
- latency alignment before subtraction/blending;
- per-source model selection;
- only after a 24 s benchmark passes, process the full 18-track song.
