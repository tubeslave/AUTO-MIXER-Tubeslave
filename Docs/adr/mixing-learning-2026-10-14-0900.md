# ADR — Separate compressor control accuracy from waveform and listening endpoints

- **ID:** `ADR-ML-2026-10-14-0900`
- **Status:** accepted for research queue; not runtime policy
- **Date:** 2026-10-14
- **Related update:** [ML-2026-10-14-0900](../mixing_learning_updates/ML-2026-10-14-0900.md)
- **Previous search:** [ML-2026-10-13-1500](../mixing_learning_updates/ML-2026-10-13-1500.md)

## Context

DAFx26 paper 30 demonstrates that output-waveform L1 and MSTE can assign similar errors to compressor models whose measured gain-reduction trajectories differ substantially. The experiment is deliberately fitted and evaluated on the same examples, has no listener test and uses a simplified compressor topology.

The associated CC-BY dataset exposes paired stereo input, hardware output and measured control trajectory. It is appropriate for metric/model research but its peak-normalised excerpts are not mix loudness references.

An official engineer companion article separately suggests that a parallel drum return can create room-like density when dedicated room microphones are absent. That is a song-specific technique, not a universal chain.

## Decision

1. For compressor-model validation, report direct gain-trajectory error separately from waveform and perceptual endpoints whenever the trajectory is observable.
2. Do not treat a lower control error as sufficient evidence of better sound, mix preference or hardware colour.
3. Require a song-disjoint held-out split before claiming generalisation.
4. Preserve topology, alignment and stereo-link parity when comparing objectives or models.
5. Treat paper hardware settings and error values as reproduction metadata, not project presets or acceptance thresholds.
6. Keep any parallel-drum ambience technique shadow-only and test one frozen return as the only changed factor.
7. Use separate raw diagnostics and BS.1770-matched, true-peak-safe blind listening copies.
8. Keep all rules `auto_apply:false` and experiments `not_run_blocked`.

## Consequences

- Compressor validation becomes a dual-endpoint workflow rather than a single waveform-loss ranking.
- The CC-BY SSL dataset is queued for manifest/split/parity audit; no files are downloaded by this update.
- Parallel drum ambience is only considered when room capture is inadequate and is cancelled for cymbal wash, pumping, lost ghost notes, stereo narrowing or masking.
- No audio, DSP, runtime, model or training change is authorised.
- The next executable task remains `EXP-ML-PENA-AB15-LISTEN-20261010-1800`.
