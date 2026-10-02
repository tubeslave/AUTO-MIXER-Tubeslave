# ADR: constrain corpus-driven reverb findings to listening-tested experiments

- **Date:** 2026-10-02
- **Update:** `ML-2026-10-02-1500`
- **Status:** accepted for research queue; not accepted for automatic application

## Context

DAFx-26 paper `DAFx26_paper_06` reports a corpus-driven six-control modal reverberator with useful engineering detail and moderate-to-strong regression fit for several damping targets. Its density fits are weaker, its extreme controls extrapolate beyond the central corpus, and it contains no formal listener evaluation. A newly reviewed Universal Audio tutorial also demonstrates pre-reverb compression and gated-snare ambience restoration, but is a single vendor-produced workflow without stated loudness matching.

## Decision

1. Add both sources and their bounded findings to Mixing Learning.
2. Keep all derived rules at `auto_apply:false`.
3. Treat the modal-reverb controls as experimental coordinates, not independent perceptual truths or production defaults.
4. Admit new reverb findings only through one-factor, randomized, BS.1770-matched project-audio tests with explicit protected qualities and cancellation criteria.
5. Keep transcript-unavailable YouTube sources as metadata-only queued items.

## Consequences

- `EXP-ML-20261002-1500-01` is queued as `not_run` behind the already executable `EXP-ML-20261001-1800-01`.
- No DSP implementation, audio processing, model training, or automatic mixing rule changes are authorized.
- The paper's linked code and the source-corpus redistribution rights remain unverified.
