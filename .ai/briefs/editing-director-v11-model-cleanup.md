# Editing Director v1.1 — Model-first DeNoise / Multi-mic DeBleed

Goal: remove noise and microphone bleed without audible pumping, chopped tails, musical-noise artifacts or timbral damage.

Hard constraints:
1. no gates, expanders, phrase masks, region attenuation or event-window muting in the de-bleed stage;
2. de-noise and de-bleed are separate problems;
3. de-noise uses a pretrained enhancement model or a fixed-profile process with dry/residual confidence blending;
4. de-bleed is treated as synchronized multi-channel source separation: all close mics and useful references are inputs, intended clean close-mic sources are outputs;
5. do not call a speech denoiser a de-bleed model;
6. every processed track exposes the removed residual for audition;
7. processing strength is controlled by residual blend, not by moving a gain gate;
8. drum phase/alignment is checked after separation;
9. guitar/bass scrape and inter-note artifacts use a dedicated artifact model/detector, not broadband attenuation;
10. accept only after RAW / CLEAN / REMOVED-RESIDUAL solo listening plus full-mix A/B.

Candidate components:
- vocals: DeepFilterNet-class full-band speech enhancement only for stationary/noise-like contamination, with bounded residual blend;
- musical bleed: multi-channel source-separation/debleed model; prefer a permutation-equivariant multi-mic architecture when weights/code are available;
- guitar/bass: model-based denoise plus localized artifact repair.

Fallback rule: if a suitable pretrained de-bleed model is unavailable, leave bleed untouched rather than substitute a gate-like heuristic.
