# ADR: block Distance Space integration and test the cue bundle against a level-only control

- **Date:** 2026-10-03
- **Update:** `ML-2026-10-03-1200`
- **Status:** accepted

## Context

The DAFx-26 paper reports a favorable 18-listener comparison for a single-control auditory-distance renderer and points to its repository for further data and analysis. Source audit confirms the core steady-state mappings, but the audited public revision contains no participant-level data, analysis script, spectral-validation test, committed benchmark result, unit tests or CI. The build file uses an author-local JUCE path. The source also reveals a GUI-dependent default-HRTF initialization path and a block-size-dependent HRTF crossfade ramp. Embedded KEMAR asset rights are not documented separately from the Apache-2.0 code licence.

## Decision

1. Keep the coupled-distance concept as a bounded production hypothesis, not a validated rule.
2. Block integration of repository code and embedded IR assets until there is a reproducible clean build, tests, explicit asset provenance/rights and a corrected headless initialization/automation path.
3. Exclude HRTF from the first project experiment.
4. Replace baseline-versus-macro testing with a stricter level-only control versus an equally attenuated coupled non-level cue bundle.
5. Preregister every derived parameter or explicitly document any frozen deviation from the paper implementation.
6. Keep every rule `auto_apply:false` and every unexecuted experiment `not_run`.

## Consequences

- Paper effect sizes remain source-reported and `citation_check:partial`; they are not reproduced by the repository audit.
- `EXP-ML-20261003-1200-01` tests whether the cue bundle adds distance/coherence beyond equal direct attenuation, with global BS.1770 matching and separate headphone/loudspeaker blocks.
- The immediately executable priority remains `EXP-ML-20261001-1800-01`, followed by the pre-reverb-compressor test.
- No DSP, runtime, audio or model was changed.
