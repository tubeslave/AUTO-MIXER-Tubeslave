# Ptitsa pretrained debleed benchmark — 2026-09-23

Human constraint: v0.9/v0.10 gate-like cleanup is rejected because it creates audible defects. Do not reintroduce it.

New candidates discovered:
- MelBand Roformer Bleed Suppressor V1 (unwa/97chris), explicitly intended as bleed suppression / post-processing.
- MelBand Roformer Denoise-Debleed (Gabox), a pretrained cleanup preprocessor.
- Mel-Roformer-Denoise-Aufr33, used as a denoise-only control.

Important scope note:
- community documentation says Gabox Denoise-Debleed is mainly for noise/bleed contamination and is not reliable for vocal-residue removal;
- Bleed Suppressor V1 is described as post-processing rather than first-pass source separation;
- therefore no candidate is accepted by name or metrics alone.

Benchmark protocol:
1. fetch short excerpts directly from the original public Yandex multitrack;
2. use raw close-mic tracks, not previous heuristic cleanup;
3. run all three pretrained models on identical excerpts;
4. produce RAW, CLEAN and REMOVED = RAW-CLEAN;
5. listen to REMOVED first: target vocal words, snare body/transient, bass notes or guitar notes in REMOVED are a rejection signal;
6. only then compare CLEAN;
7. no full-song processing until one model/strength passes the short benchmark.

Current workflow:
.github/workflows/debleed-benchmark.yml

Source:
https://disk.yandex.ru/d/OvdRlysTGFbxEQ
Ptitsa source range begins at 29:54 in the long synchronized recording.
