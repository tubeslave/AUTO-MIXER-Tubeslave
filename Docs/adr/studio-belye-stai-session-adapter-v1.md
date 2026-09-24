# ADR: Belye Stai Session Renderer Adapter v1

## Context

Routed Contribution Renderer v1 requires an authoritative deterministic `SessionRender` callback. The delivered Belye Stai recipe was still a procedural `mix_dsp.py` with hard-coded project I/O. A dry-source substitution is not exact because the recipe contains lead-vocal-driven guitar lift and masking, kick-linked bass ducking, send/return effects, a shared nonlinear mix-glue compressor, common trim, and a documented drum-accent refinement.

## Bounded task

Migrate the delivered conventional-DSP graph into an in-memory callable which accepts immutable raw source overrides and returns `SessionRender`. Before using it for a new compression hypothesis, prove that a no-change render reproduces the stored Belye Stai premaster. No mastering, learned audio processing, live/backend code, paid service or subjective baseline promotion is in scope.

## Decision

`audio_workbench.belye_stai_session.BelyeStaiSessionRenderer` owns all 18 raw source arrays, copies every override, executes the delivered mix order in memory, and returns mix/vocal/drums/room anchors. The original processing coefficients and order are retained. The renderer also includes the later documented bounded refinement from `refine_accent.py`: a linked 3 dB Gaussian DRUMS-bus reduction centered at 159.056 s with sigma 55 ms. That refinement is part of the authoritative delivered premaster graph, not mastering.

The session callback deliberately returns float PCM and does not write files. The historical PCM24 export can be reproduced separately with the documented triangular dither seed `20260924`.

## Experiment and revision

The first real full-song migration rendered the base `mix_dsp.py` graph only. It matched the stored delivery essentially everywhere outside the break accent, but did **not** reproduce the saved premaster: the stored file contains the subsequent `refine_accent.py` stage. The implementation was therefore revised rather than accepting a loose tolerance.

After incorporating that documented stage, the adapter rendered all 18 original 24-bit/44.1 kHz WAVs for all 9,128,700 frames. Applying the documented PCM24 dither seed produced a file byte-identical to the stored premaster:

- stored premaster SHA-256: `9cf6c5a9ffbd02d7685d6f16e76dd98525662499ae33364277d1c75ed39c8e76`
- adapter premaster SHA-256: `9cf6c5a9ffbd02d7685d6f16e76dd98525662499ae33364277d1c75ed39c8e76`
- decoded PCM max absolute error: `0.0`
- decoded PCM RMSE: `0.0`
- equal samples: `100%`
- full in-memory graph render observed in the validation environment: about 55 s after source loading

All 18 source file hashes were rechecked against the delivered manifest before the real render. The source files were not modified.

## CI plan

Repository tests use deterministic short synthetic 18-source fixtures because the real multitrack is not stored in Git. They verify repeated render identity, required SessionRender anchors, source immutability, a real source override propagating through the graph, and fail-closed source/layout validation. The exact full-song no-change proof remains external experiment evidence tied to the original source hashes and stored premaster hash.

## Acceptance boundary

Byte-identical no-change reproduction validates the migrated graph for this environment and source set. It does not prove musical superiority of any future candidate. Compression candidates must still pass Routed Contribution / Perceptual Critic objective gates and level-matched human listening. No baseline may be promoted by this adapter.

## Next task

Run one bounded Compression Director candidate family through this exact full-session renderer, use full-mix Perceptual Critic evidence to reject unsafe candidates, and export only level-matched survivors for human A/B. Optimize stage caching only after correctness is retained by the same no-change identity test.
