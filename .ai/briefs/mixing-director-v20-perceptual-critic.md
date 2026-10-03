# Mixing Director v2.0 — Perceptual Mix Critic

This follows the Instrument Directors and Autonomous Iteration Loop.

The critic does NOT output a single quality score. It emits interpretable perceptual diagnostics and bounded hypotheses.

Dimensions:
- foreground/background relationship;
- vocal intelligibility against competing music;
- drum punch;
- density;
- harshness;
- depth / room audibility;
- stereo width;
- climax lift across sections.

Architecture:
1. deterministic signal features provide explainable measurements;
2. optional learned audio embeddings / music-quality models may be adapters, never sole judges;
3. critic emits one hypothesis at a time;
4. Autonomous Iteration renders candidate B;
5. target dimension must improve while loudness, spectrum, phase and previously accepted dimensions remain guarded;
6. RAW/CLEAN residual rules from Editing Cleanup remain separate from Mix Critic;
7. no overall "mix quality 8/10" metric;
8. during development every accepted perceptual change still requires level-matched human A/B.

Next adapters to evaluate:
- MuQ / music embeddings for section/context similarity;
- learned audio-quality predictor for artifact detection;
- stem-aware foreground/background classifier;
- reference-free section climax/context model.

The deterministic critic is intentionally modest: it measures evidence, it does not pretend to hear like a human.
